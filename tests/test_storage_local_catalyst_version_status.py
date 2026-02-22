"""
Unit tests for catalyst_version_status and catalyst_version_info functions
Uses real git repositories in temporary folders for integration testing
"""

import subprocess
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from overity.storage.local import LocalStorage
from overity.model.versioning import VersioningStatus
from overity.errors import NoVersionAvailable


def run_git(args, cwd=None):
    """Helper to run git commands"""
    result = subprocess.run(
        ["git"] + args, cwd=str(cwd) if cwd else None, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"git command failed: {result.stderr}")
    return result


def create_git_repo(path, with_commit=True):
    """Create a git repository at the given path"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    run_git(["init"], cwd=path)
    run_git(["config", "user.email", "test@test.com"], cwd=path)
    run_git(["config", "user.name", "Test User"], cwd=path)

    if with_commit:
        # Create initial commit
        (path / "README.md").write_text("# Test repo")
        run_git(["add", "README.md"], cwd=path)
        run_git(["commit", "-m", "Initial commit"], cwd=path)

    return path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


class TestCatalystVersionStatusIntegration:
    """Integration tests for catalyst_version_status with real git repos"""

    def test_clean_no_changes_real_repo(self, temp_dir):
        """Test when catalyst folder is clean in a real git repo"""
        # Create a repo with catalyst folder
        create_git_repo(temp_dir)
        catalyst_dir = temp_dir / "catalyst"
        catalyst_dir.mkdir()
        (catalyst_dir / "bench.toml").write_text("[bench]")
        run_git(["add", "."], cwd=temp_dir)
        run_git(["commit", "-m", "Add catalyst"], cwd=temp_dir)

        storage = LocalStorage(temp_dir)
        result = storage.catalyst_version_status()

        assert result == VersioningStatus.Clean

    def test_dirty_with_uncommitted_changes_real_repo(self, temp_dir):
        """Test when there are uncommitted changes in catalyst"""
        # Create a repo with catalyst folder
        create_git_repo(temp_dir)
        catalyst_dir = temp_dir / "catalyst"
        catalyst_dir.mkdir()
        (catalyst_dir / "bench.toml").write_text("[bench]")
        run_git(["add", "."], cwd=temp_dir)
        run_git(["commit", "-m", "Add catalyst"], cwd=temp_dir)

        # Add uncommitted change
        (catalyst_dir / "bench.toml").write_text('[bench]\nname = "test"')

        storage = LocalStorage(temp_dir)
        result = storage.catalyst_version_status()

        assert result == VersioningStatus.Dirty

    def test_clean_changes_outside_catalyst_real_repo(self, temp_dir):
        """Test that changes outside catalyst don't affect status"""
        create_git_repo(temp_dir)

        # Create catalyst folder
        catalyst_dir = temp_dir / "catalyst"
        catalyst_dir.mkdir()
        (catalyst_dir / "bench.toml").write_text("[bench]")
        run_git(["add", "."], cwd=temp_dir)
        run_git(["commit", "-m", "Add catalyst"], cwd=temp_dir)

        # Add file outside catalyst
        (temp_dir / "other.txt").write_text("other content")
        run_git(["add", "other.txt"], cwd=temp_dir)

        storage = LocalStorage(temp_dir)
        result = storage.catalyst_version_status()

        assert result == VersioningStatus.Clean

    def test_not_versioned_no_git_repo_real(self, temp_dir):
        """Test when there is no git repository"""
        # Just create a folder without git
        (temp_dir / "catalyst").mkdir()
        (temp_dir / "catalyst" / "file.txt").write_text("content")

        storage = LocalStorage(temp_dir)
        result = storage.catalyst_version_status()

        assert result == VersioningStatus.NotVersioned

    def test_catalyst_is_git_submodule_clean_real(self, temp_dir):
        """Test when catalyst folder is a git submodule (clean)"""
        # Create parent repo
        create_git_repo(temp_dir)

        # Create catalyst as a separate repo (will be submodule)
        catalyst_dir = temp_dir / "catalyst"
        create_git_repo(catalyst_dir)
        (catalyst_dir / "bench.toml").write_text("[bench]")
        run_git(["add", "."], cwd=catalyst_dir)
        run_git(["commit", "-m", "Add bench"], cwd=catalyst_dir)

        # Add the submodule folder to parent repo
        run_git(["add", "catalyst"], cwd=temp_dir)
        run_git(["commit", "-m", "Add catalyst submodule"], cwd=temp_dir)

        storage = LocalStorage(temp_dir)
        result = storage.catalyst_version_status()

        assert result == VersioningStatus.Clean

    def test_catalyst_is_git_submodule_dirty_real(self, temp_dir):
        """Test when catalyst folder is a git submodule (dirty)"""
        # Create parent repo
        create_git_repo(temp_dir)

        # Create catalyst as a separate repo (will be submodule)
        catalyst_dir = temp_dir / "catalyst"
        create_git_repo(catalyst_dir)
        (catalyst_dir / "bench.toml").write_text("[bench]")
        run_git(["add", "."], cwd=catalyst_dir)
        run_git(["commit", "-m", "Add bench"], cwd=catalyst_dir)

        # Make uncommitted change
        (catalyst_dir / "bench.toml").write_text('[bench]\nname = "modified"')

        storage = LocalStorage(temp_dir)
        result = storage.catalyst_version_status()

        assert result == VersioningStatus.Dirty


class TestCatalystVersionInfoIntegration:
    """Integration tests for catalyst_version_info with real git repos"""

    def test_returns_real_commit_hash(self, temp_dir):
        """Test returning an actual git commit hash"""
        create_git_repo(temp_dir)
        (temp_dir / "file.txt").write_text("content")
        run_git(["add", "."], cwd=temp_dir)
        run_git(["commit", "-m", "Initial commit"], cwd=temp_dir)

        # Get the actual commit hash
        result = run_git(["rev-parse", "HEAD"], cwd=temp_dir)
        expected_hash = result.stdout.strip()

        storage = LocalStorage(temp_dir)
        version_info = storage.catalyst_version_info()

        assert version_info == expected_hash
        assert len(version_info) == 40  # SHA-1 hash length

    def test_returns_different_hash_after_new_commit(self, temp_dir):
        """Test that new commits return different hashes"""
        create_git_repo(temp_dir)
        (temp_dir / "file.txt").write_text("content")
        run_git(["add", "."], cwd=temp_dir)
        run_git(["commit", "-m", "First commit"], cwd=temp_dir)

        storage = LocalStorage(temp_dir)
        hash1 = storage.catalyst_version_info()

        # Make another commit
        (temp_dir / "file.txt").write_text("modified content")
        run_git(["add", "."], cwd=temp_dir)
        run_git(["commit", "-m", "Second commit"], cwd=temp_dir)

        hash2 = storage.catalyst_version_info()

        assert hash1 != hash2
        assert len(hash1) == 40
        assert len(hash2) == 40

    def test_raises_no_version_available_no_git_real(self, temp_dir):
        """Test raising NoVersionAvailable when no git repo exists"""
        # Create a folder without git
        (temp_dir / "catalyst").mkdir()

        storage = LocalStorage(temp_dir)
        with pytest.raises(NoVersionAvailable):
            storage.catalyst_version_info()

    def test_raises_no_version_available_no_commits_real(self, temp_dir):
        """Test raising NoVersionAvailable when repo has no commits"""
        # Create repo but don't make any commits
        create_git_repo(temp_dir, with_commit=False)

        storage = LocalStorage(temp_dir)
        with pytest.raises(NoVersionAvailable):
            storage.catalyst_version_info()

    def test_returns_commit_from_parent_repo_when_catalyst_is_submodule(self, temp_dir):
        """Test that version info comes from parent repo, not submodule"""
        # Create parent repo
        create_git_repo(temp_dir)

        # Create catalyst as a submodule with its own repo
        catalyst_dir = temp_dir / "catalyst"
        create_git_repo(catalyst_dir)
        (catalyst_dir / "bench.toml").write_text("[bench]")
        run_git(["add", "."], cwd=catalyst_dir)
        run_git(["commit", "-m", "Add bench"], cwd=catalyst_dir)

        # Add catalyst to parent and commit
        run_git(["add", "catalyst"], cwd=temp_dir)
        run_git(["commit", "-m", "Add catalyst submodule"], cwd=temp_dir)

        # Get the parent repo commit hash (not the submodule's)
        result = run_git(["rev-parse", "HEAD"], cwd=temp_dir)
        expected_parent_hash = result.stdout.strip()

        storage = LocalStorage(temp_dir)
        version_info = storage.catalyst_version_info()

        # The implementation uses base_folder, so it returns the parent's commit
        assert version_info == expected_parent_hash

        # Verify it's different from the submodule's commit
        submodule_hash = run_git(["rev-parse", "HEAD"], cwd=catalyst_dir).stdout.strip()
        assert version_info != submodule_hash


# Unit tests for edge cases using mocks
class TestCatalystVersionStatus:
    """Unit tests for catalyst_version_status function"""

    def test_not_versioned_no_git_repo(self):
        """Test when there is no git repository"""
        with patch("overity.storage.local.git_utils.nearest_repo") as mock_nearest:
            mock_nearest.return_value = None

            storage = LocalStorage(Path("/test/program"))
            result = storage.catalyst_version_status()

            assert result == VersioningStatus.NotVersioned
            mock_nearest.assert_called_once_with(storage.base_folder)

    def test_clean_no_changes(self):
        """Test when catalyst folder is clean (no changes)"""
        with patch(
            "overity.storage.local.git_utils.nearest_repo"
        ) as mock_nearest, patch(
            "overity.storage.local.git_utils.git_status"
        ) as mock_git_status:
            # Mock that we found a repo at the catalyst folder itself
            mock_nearest.return_value = Path("/test/program/catalyst")

            # No changes in git status
            mock_git_status.return_value = []

            storage = LocalStorage(Path("/test/program"))
            result = storage.catalyst_version_status()

            assert result == VersioningStatus.Clean
            mock_nearest.assert_called_once_with(storage.base_folder)
            mock_git_status.assert_called_once_with(str(Path("/test/program/catalyst")))

    def test_clean_changes_outside_catalyst(self):
        """Test when there are changes but outside catalyst folder"""
        with patch(
            "overity.storage.local.git_utils.nearest_repo"
        ) as mock_nearest, patch(
            "overity.storage.local.git_utils.git_status"
        ) as mock_git_status, patch(
            "overity.storage.local.path_utils.is_subpath"
        ) as mock_is_subpath:
            mock_nearest.return_value = Path("/test/program")

            # Create mock changes outside catalyst folder
            mock_change = MagicMock()
            mock_change.from_path = "other_folder/file.txt"
            mock_git_status.return_value = [mock_change]

            # is_subpath returns False for changes outside catalyst
            mock_is_subpath.return_value = False

            storage = LocalStorage(Path("/test/program"))
            result = storage.catalyst_version_status()

            assert result == VersioningStatus.Clean

    def test_dirty_changes_inside_catalyst(self):
        """Test when there are changes inside catalyst folder"""
        with patch(
            "overity.storage.local.git_utils.nearest_repo"
        ) as mock_nearest, patch(
            "overity.storage.local.git_utils.git_status"
        ) as mock_git_status, patch(
            "overity.storage.local.path_utils.is_subpath"
        ) as mock_is_subpath:
            mock_nearest.return_value = Path("/test/program")

            # Create mock changes inside catalyst folder
            mock_change = MagicMock()
            mock_change.from_path = "catalyst/bench.toml"
            mock_git_status.return_value = [mock_change]

            # is_subpath returns True for changes inside catalyst
            mock_is_subpath.return_value = True

            storage = LocalStorage(Path("/test/program"))
            result = storage.catalyst_version_status()

            assert result == VersioningStatus.Dirty

    def test_catalyst_not_under_repo(self):
        """Test when catalyst folder is not under the found repo"""
        with patch(
            "overity.storage.local.git_utils.nearest_repo"
        ) as mock_nearest, patch(
            "overity.storage.local.git_utils.git_status"
        ) as mock_git_status:
            # The repo is at a completely different location from catalyst
            mock_nearest.return_value = Path("/other/repo")

            storage = LocalStorage(Path("/test/program"))

            # The catalyst folder is /test/program/catalyst, which is not under /other/repo
            # So catalyst.relative_to(/other/repo) should raise ValueError
            try:
                storage.catalyst_folder.relative_to(Path("/other/repo"))
                # If we get here, the paths happen to work (depends on filesystem)
                result = storage.catalyst_version_status()
                assert result in [VersioningStatus.Clean, VersioningStatus.Dirty]
            except ValueError:
                # This is the expected case - catalyst is not under the repo
                result = storage.catalyst_version_status()
                assert result == VersioningStatus.NotVersioned
                # git_status should not be called since catalyst is not under repo
                mock_git_status.assert_not_called()


class TestCatalystVersionInfo:
    """Unit tests for catalyst_version_info function"""

    def test_raises_no_version_available_no_git(self):
        """Test raising NoVersionAvailable when git command fails"""
        with patch(
            "overity.storage.local.git_utils.current_commit"
        ) as mock_current_commit:
            mock_current_commit.side_effect = RuntimeError("git command not found")

            storage = LocalStorage(Path("/test/program"))
            with pytest.raises(NoVersionAvailable) as exc_info:
                storage.catalyst_version_info()

            assert "catalyst" in str(exc_info.value)
            assert "/test/program/catalyst" in str(exc_info.value)

    def test_path_passed_correctly_to_current_commit(self):
        """Test that catalyst folder path is passed correctly to current_commit"""
        with patch(
            "overity.storage.local.git_utils.current_commit"
        ) as mock_current_commit:
            mock_current_commit.return_value = "abc123" + "0" * 36

            storage = LocalStorage(Path("/some/deeply/nested/program/path"))
            storage.catalyst_version_info()

            # Verify the correct path was passed
            mock_current_commit.assert_called_once()
            call_args = mock_current_commit.call_args[0][0]
            assert str(call_args) == "/some/deeply/nested/program/path"
