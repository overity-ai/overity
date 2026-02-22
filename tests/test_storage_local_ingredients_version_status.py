"""
Unit tests for ingredients_version_status and ingredients_version_info functions
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


def create_git_submodule(parent_path, submodule_name, with_commit=True):
    """Create a git submodule within a parent repo"""
    submodule_path = parent_path / submodule_name
    create_git_repo(submodule_path, with_commit=with_commit)

    # Add as submodule to parent
    run_git(["submodule", "add", str(submodule_path), submodule_name], cwd=parent_path)
    run_git(["commit", "-m", f"Add {submodule_name} submodule"], cwd=parent_path)

    return submodule_path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


class TestIngredientsVersionStatusIntegration:
    """Integration tests for ingredients_version_status with real git repos"""

    def test_clean_no_changes_real_repo(self, temp_dir):
        """Test when ingredients folder is clean in a real git repo"""
        # Create a repo with ingredients folder
        create_git_repo(temp_dir)
        ingredients_dir = temp_dir / "ingredients"
        ingredients_dir.mkdir()
        (ingredients_dir / "method.py").write_text("# method")
        run_git(["add", "."], cwd=temp_dir)
        run_git(["commit", "-m", "Add ingredients"], cwd=temp_dir)

        storage = LocalStorage(temp_dir)
        result = storage.ingredients_version_status()

        assert result == VersioningStatus.Clean

    def test_dirty_with_uncommitted_changes_real_repo(self, temp_dir):
        """Test when there are uncommitted changes in ingredients"""
        # Create a repo with ingredients folder
        create_git_repo(temp_dir)
        ingredients_dir = temp_dir / "ingredients"
        ingredients_dir.mkdir()
        (ingredients_dir / "method.py").write_text("# method")
        run_git(["add", "."], cwd=temp_dir)
        run_git(["commit", "-m", "Add ingredients"], cwd=temp_dir)

        # Add uncommitted change
        (ingredients_dir / "method.py").write_text("# modified method")

        storage = LocalStorage(temp_dir)
        result = storage.ingredients_version_status()

        assert result == VersioningStatus.Dirty

    def test_clean_changes_outside_ingredients_real_repo(self, temp_dir):
        """Test that changes outside ingredients don't affect status"""
        create_git_repo(temp_dir)

        # Create ingredients folder
        ingredients_dir = temp_dir / "ingredients"
        ingredients_dir.mkdir()
        (ingredients_dir / "method.py").write_text("# method")
        run_git(["add", "."], cwd=temp_dir)
        run_git(["commit", "-m", "Add ingredients"], cwd=temp_dir)

        # Add file outside ingredients
        (temp_dir / "other.txt").write_text("other content")
        run_git(["add", "other.txt"], cwd=temp_dir)

        storage = LocalStorage(temp_dir)
        result = storage.ingredients_version_status()

        assert result == VersioningStatus.Clean

    def test_not_versioned_no_git_repo_real(self, temp_dir):
        """Test when there is no git repository"""
        # Just create a folder without git
        (temp_dir / "ingredients").mkdir()
        (temp_dir / "ingredients" / "file.txt").write_text("content")

        storage = LocalStorage(temp_dir)
        result = storage.ingredients_version_status()

        assert result == VersioningStatus.NotVersioned

    def test_ingredients_is_git_submodule_clean_real(self, temp_dir):
        """Test when ingredients folder is a git submodule (clean)"""
        # Create parent repo
        create_git_repo(temp_dir)

        # Create ingredients as a separate repo (will be submodule)
        ingredients_dir = temp_dir / "ingredients"
        create_git_repo(ingredients_dir)
        (ingredients_dir / "method.py").write_text("# method")
        run_git(["add", "."], cwd=ingredients_dir)
        run_git(["commit", "-m", "Add method"], cwd=ingredients_dir)

        # Add the submodule folder to parent repo
        # Note: We don't use git submodule add here, just track the folder
        # The ingredients folder itself is a separate repo
        # For the test to be clean, we need to commit the submodule reference
        run_git(["add", "ingredients"], cwd=temp_dir)
        run_git(["commit", "-m", "Add ingredients submodule"], cwd=temp_dir)

        storage = LocalStorage(temp_dir)
        result = storage.ingredients_version_status()

        assert result == VersioningStatus.Clean

    def test_ingredients_is_git_submodule_dirty_real(self, temp_dir):
        """Test when ingredients folder is a git submodule (dirty)"""
        # Create parent repo
        create_git_repo(temp_dir)

        # Create ingredients as a separate repo (will be submodule)
        ingredients_dir = temp_dir / "ingredients"
        create_git_repo(ingredients_dir)
        (ingredients_dir / "method.py").write_text("# method")
        run_git(["add", "."], cwd=ingredients_dir)
        run_git(["commit", "-m", "Add method"], cwd=ingredients_dir)

        # Make uncommitted change
        (ingredients_dir / "method.py").write_text("# modified method")

        storage = LocalStorage(temp_dir)
        result = storage.ingredients_version_status()

        assert result == VersioningStatus.Dirty


class TestIngredientsVersionInfoIntegration:
    """Integration tests for ingredients_version_info with real git repos"""

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
        version_info = storage.ingredients_version_info()

        assert version_info == expected_hash
        assert len(version_info) == 40  # SHA-1 hash length

    def test_returns_different_hash_after_new_commit(self, temp_dir):
        """Test that new commits return different hashes"""
        create_git_repo(temp_dir)
        (temp_dir / "file.txt").write_text("content")
        run_git(["add", "."], cwd=temp_dir)
        run_git(["commit", "-m", "First commit"], cwd=temp_dir)

        storage = LocalStorage(temp_dir)
        hash1 = storage.ingredients_version_info()

        # Make another commit
        (temp_dir / "file.txt").write_text("modified content")
        run_git(["add", "."], cwd=temp_dir)
        run_git(["commit", "-m", "Second commit"], cwd=temp_dir)

        hash2 = storage.ingredients_version_info()

        assert hash1 != hash2
        assert len(hash1) == 40
        assert len(hash2) == 40

    def test_raises_no_version_available_no_git_real(self, temp_dir):
        """Test raising NoVersionAvailable when no git repo exists"""
        # Create a folder without git
        (temp_dir / "ingredients").mkdir()

        storage = LocalStorage(temp_dir)
        with pytest.raises(NoVersionAvailable):
            storage.ingredients_version_info()

    def test_raises_no_version_available_no_commits_real(self, temp_dir):
        """Test raising NoVersionAvailable when repo has no commits"""
        # Create repo but don't make any commits
        create_git_repo(temp_dir, with_commit=False)

        storage = LocalStorage(temp_dir)
        with pytest.raises(NoVersionAvailable):
            storage.ingredients_version_info()

    def test_returns_commit_from_parent_repo_when_ingredients_is_submodule(
        self, temp_dir
    ):
        """Test that version info comes from parent repo, not submodule"""
        # Create parent repo
        create_git_repo(temp_dir)

        # Create ingredients as a submodule with its own repo
        ingredients_dir = temp_dir / "ingredients"
        create_git_repo(ingredients_dir)
        (ingredients_dir / "method.py").write_text("# method")
        run_git(["add", "."], cwd=ingredients_dir)
        run_git(["commit", "-m", "Add method"], cwd=ingredients_dir)

        # Add ingredients to parent and commit
        run_git(["add", "ingredients"], cwd=temp_dir)
        run_git(["commit", "-m", "Add ingredients submodule"], cwd=temp_dir)

        # Get the parent repo commit hash (not the submodule's)
        result = run_git(["rev-parse", "HEAD"], cwd=temp_dir)
        expected_parent_hash = result.stdout.strip()

        storage = LocalStorage(temp_dir)
        version_info = storage.ingredients_version_info()

        # The implementation uses base_folder, so it returns the parent's commit
        assert version_info == expected_parent_hash

        # Verify it's different from the submodule's commit
        submodule_hash = run_git(
            ["rev-parse", "HEAD"], cwd=ingredients_dir
        ).stdout.strip()
        assert version_info != submodule_hash


# Keep existing mocked unit tests for edge cases
class TestIngredientsVersionStatus:
    """Unit tests for ingredients_version_status function"""

    def test_not_versioned_no_git_repo(self):
        """Test when there is no git repository"""
        with patch("overity.storage.local.git_utils.nearest_repo") as mock_nearest:
            mock_nearest.return_value = None

            storage = LocalStorage(Path("/test/program"))
            result = storage.ingredients_version_status()

            assert result == VersioningStatus.NotVersioned
            mock_nearest.assert_called_once_with(storage.base_folder)

    def test_clean_no_changes(self):
        """Test when ingredients folder is clean (no changes)"""
        with patch(
            "overity.storage.local.git_utils.nearest_repo"
        ) as mock_nearest, patch(
            "overity.storage.local.git_utils.git_status"
        ) as mock_git_status:
            # Mock that we found a repo at the ingredients folder itself
            mock_nearest.return_value = Path("/test/program/ingredients")

            # No changes in git status
            mock_git_status.return_value = []

            storage = LocalStorage(Path("/test/program"))
            result = storage.ingredients_version_status()

            assert result == VersioningStatus.Clean
            mock_nearest.assert_called_once_with(storage.base_folder)
            mock_git_status.assert_called_once_with(
                str(Path("/test/program/ingredients"))
            )

    def test_clean_changes_outside_ingredients(self):
        """Test when there are changes but outside ingredients folder"""
        with patch(
            "overity.storage.local.git_utils.nearest_repo"
        ) as mock_nearest, patch(
            "overity.storage.local.git_utils.git_status"
        ) as mock_git_status, patch(
            "overity.storage.local.path_utils.is_subpath"
        ) as mock_is_subpath:
            mock_nearest.return_value = Path("/test/program")

            # Create mock changes outside ingredients folder
            mock_change = MagicMock()
            mock_change.from_path = "other_folder/file.txt"
            mock_git_status.return_value = [mock_change]

            # is_subpath returns False for changes outside ingredients
            mock_is_subpath.return_value = False

            storage = LocalStorage(Path("/test/program"))
            result = storage.ingredients_version_status()

            assert result == VersioningStatus.Clean

    def test_dirty_changes_inside_ingredients(self):
        """Test when there are changes inside ingredients folder"""
        with patch(
            "overity.storage.local.git_utils.nearest_repo"
        ) as mock_nearest, patch(
            "overity.storage.local.git_utils.git_status"
        ) as mock_git_status, patch(
            "overity.storage.local.path_utils.is_subpath"
        ) as mock_is_subpath:
            mock_nearest.return_value = Path("/test/program")

            # Create mock changes inside ingredients folder
            mock_change = MagicMock()
            mock_change.from_path = "ingredients/training_optimization/method.py"
            mock_git_status.return_value = [mock_change]

            # is_subpath returns True for changes inside ingredients
            mock_is_subpath.return_value = True

            storage = LocalStorage(Path("/test/program"))
            result = storage.ingredients_version_status()

            assert result == VersioningStatus.Dirty

    def test_ingredients_not_under_repo(self):
        """Test when ingredients folder is not under the found repo"""
        with patch(
            "overity.storage.local.git_utils.nearest_repo"
        ) as mock_nearest, patch(
            "overity.storage.local.git_utils.git_status"
        ) as mock_git_status:
            # The repo is at a completely different location from ingredients
            mock_nearest.return_value = Path("/other/repo")

            storage = LocalStorage(Path("/test/program"))

            # The ingredients folder is /test/program/ingredients, which is not under /other/repo
            # So ingredients.relative_to(/other/repo) should raise ValueError
            # We test this by checking the actual behavior
            try:
                storage.ingredients_folder.relative_to(Path("/other/repo"))
                # If we get here, the paths happen to work (depends on filesystem)
                # In that case, the test will call git_status, which is fine
                # Just verify we get a valid result
                result = storage.ingredients_version_status()
                assert result in [VersioningStatus.Clean, VersioningStatus.Dirty]
            except ValueError:
                # This is the expected case - ingredients is not under the repo
                result = storage.ingredients_version_status()
                assert result == VersioningStatus.NotVersioned
                # git_status should not be called since ingredients is not under repo
                mock_git_status.assert_not_called()


class TestIngredientsVersionInfo:
    """Unit tests for ingredients_version_info function"""

    def test_raises_no_version_available_no_git(self):
        """Test raising NoVersionAvailable when git command fails"""
        with patch(
            "overity.storage.local.git_utils.current_commit"
        ) as mock_current_commit:
            mock_current_commit.side_effect = RuntimeError("git command not found")

            storage = LocalStorage(Path("/test/program"))
            with pytest.raises(NoVersionAvailable) as exc_info:
                storage.ingredients_version_info()

            assert "ingredients" in str(exc_info.value)
            assert "/test/program/ingredients" in str(exc_info.value)

    def test_path_passed_correctly_to_current_commit(self):
        """Test that ingredients folder path is passed correctly to current_commit"""
        with patch(
            "overity.storage.local.git_utils.current_commit"
        ) as mock_current_commit:
            mock_current_commit.return_value = "abc123" + "0" * 36

            storage = LocalStorage(Path("/some/deeply/nested/program/path"))
            storage.ingredients_version_info()

            # Verify the correct path was passed
            mock_current_commit.assert_called_once()
            call_args = mock_current_commit.call_args[0][0]
            assert str(call_args) == "/some/deeply/nested/program/path"
