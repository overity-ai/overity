"""
Unit tests for utils/git.py
"""

import pytest
import tempfile
import subprocess
from pathlib import Path

from overity.utils.git import (
    GitStatusEntryKind,
    GitStatusEntry,
    git_status,
    nearest_repo,
    farthest_repo,
    current_commit,
)


class TestCharToKind:
    """Tests for GitStatusEntryKind.try_from_str function"""

    def test_added(self):
        """Test 'A' maps to Added"""
        assert GitStatusEntryKind.try_from_str("A") == GitStatusEntryKind.Added

    def test_modified(self):
        """Test 'M' maps to Modified"""
        assert GitStatusEntryKind.try_from_str("M") == GitStatusEntryKind.Modified

    def test_deleted(self):
        """Test 'D' maps to Deleted"""
        assert GitStatusEntryKind.try_from_str("D") == GitStatusEntryKind.Deleted

    def test_renamed(self):
        """Test 'R' maps to Renamed"""
        assert GitStatusEntryKind.try_from_str("R") == GitStatusEntryKind.Renamed

    def test_copied(self):
        """Test 'C' maps to Copied"""
        assert GitStatusEntryKind.try_from_str("C") == GitStatusEntryKind.Copied

    def test_untracked(self):
        """Test '?' maps to Untracked"""
        assert GitStatusEntryKind.try_from_str("?") == GitStatusEntryKind.Untracked

    def test_ignored(self):
        """Test '!' maps to Ignored"""
        assert GitStatusEntryKind.try_from_str("!") == GitStatusEntryKind.Ignored

    def test_updated_but_unmerged(self):
        """Test 'U' maps to UpdatedButUnmerged"""
        assert (
            GitStatusEntryKind.try_from_str("U")
            == GitStatusEntryKind.UpdatedButUnmerged
        )

    def test_invalid_char(self):
        """Test invalid character returns None"""
        assert GitStatusEntryKind.try_from_str("X") is None
        assert GitStatusEntryKind.try_from_str(" ") is None
        assert GitStatusEntryKind.try_from_str("") is None

    def test_space_char(self):
        """Test space character returns None"""
        assert GitStatusEntryKind.try_from_str(" ") is None


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        # Configure git user for commits
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        yield repo_path


class TestGitStatus:
    """Tests for git_status function"""

    def test_clean_repo(self, temp_git_repo):
        """Test that a clean repo returns empty list"""
        result = git_status(str(temp_git_repo))
        assert result == []

    def test_untracked_file(self, temp_git_repo):
        """Test detecting untracked files"""
        # Create an untracked file
        (temp_git_repo / "untracked.txt").write_text("hello")

        result = git_status(str(temp_git_repo))

        assert len(result) == 1
        assert result[0].from_path == "untracked.txt"
        assert result[0].staged_kind is None
        assert result[0].unstaged_kind == GitStatusEntryKind.Untracked
        assert result[0].to_path is None

    def test_staged_file(self, temp_git_repo):
        """Test detecting staged files"""
        # Create and stage a file
        (temp_git_repo / "staged.txt").write_text("hello")
        subprocess.run(
            ["git", "add", "staged.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        result = git_status(str(temp_git_repo))

        assert len(result) == 1
        assert result[0].from_path == "staged.txt"
        assert result[0].staged_kind == GitStatusEntryKind.Added
        assert result[0].unstaged_kind is None

    def test_modified_file(self, temp_git_repo):
        """Test detecting modified files"""
        # Create, commit, then modify a file
        (temp_git_repo / "modified.txt").write_text("original")
        subprocess.run(
            ["git", "add", "modified.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        (temp_git_repo / "modified.txt").write_text("modified")

        result = git_status(str(temp_git_repo))

        assert len(result) == 1
        assert result[0].from_path == "modified.txt"
        assert result[0].staged_kind is None
        assert result[0].unstaged_kind == GitStatusEntryKind.Modified

    def test_staged_and_modified(self, temp_git_repo):
        """Test detecting staged and modified files (MM status)"""
        # Create, commit, then modify a file
        (temp_git_repo / "both.txt").write_text("original")
        subprocess.run(
            ["git", "add", "both.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        # Stage and then modify
        (temp_git_repo / "both.txt").write_text("staged version")
        subprocess.run(
            ["git", "add", "both.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        (temp_git_repo / "both.txt").write_text("modified after stage")

        result = git_status(str(temp_git_repo))

        assert len(result) == 1
        assert result[0].from_path == "both.txt"
        assert result[0].staged_kind == GitStatusEntryKind.Modified
        assert result[0].unstaged_kind == GitStatusEntryKind.Modified

    def test_deleted_file(self, temp_git_repo):
        """Test detecting deleted files"""
        # Create and commit a file, then delete it
        (temp_git_repo / "deleted.txt").write_text("to be deleted")
        subprocess.run(
            ["git", "add", "deleted.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        (temp_git_repo / "deleted.txt").unlink()

        result = git_status(str(temp_git_repo))

        assert len(result) == 1
        assert result[0].from_path == "deleted.txt"
        assert result[0].staged_kind is None
        assert result[0].unstaged_kind == GitStatusEntryKind.Deleted

    def test_staged_deleted_file(self, temp_git_repo):
        """Test detecting staged deleted files"""
        # Create, commit, then stage deletion
        (temp_git_repo / "staged_del.txt").write_text("to be deleted")
        subprocess.run(
            ["git", "add", "staged_del.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        (temp_git_repo / "staged_del.txt").unlink()
        subprocess.run(
            ["git", "add", "staged_del.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        result = git_status(str(temp_git_repo))

        assert len(result) == 1
        assert result[0].from_path == "staged_del.txt"
        assert result[0].staged_kind == GitStatusEntryKind.Deleted
        assert result[0].unstaged_kind is None

    def test_renamed_file(self, temp_git_repo):
        """Test detecting renamed files"""
        # Create and commit a file, then rename it
        (temp_git_repo / "old_name.txt").write_text("content")
        subprocess.run(
            ["git", "add", "old_name.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "mv", "old_name.txt", "new_name.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        result = git_status(str(temp_git_repo))

        assert len(result) == 1
        assert result[0].from_path == "old_name.txt"
        assert result[0].to_path == "new_name.txt"
        assert result[0].staged_kind == GitStatusEntryKind.Renamed

    def test_multiple_files(self, temp_git_repo):
        """Test detecting multiple files with different statuses"""
        # Create multiple files
        (temp_git_repo / "file1.txt").write_text("file1")
        (temp_git_repo / "file2.txt").write_text("file2")

        # Stage file1
        subprocess.run(
            ["git", "add", "file1.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        result = git_status(str(temp_git_repo))

        assert len(result) == 2
        paths = {entry.from_path for entry in result}
        assert paths == {"file1.txt", "file2.txt"}

    def test_nonexistent_path(self):
        """Test that non-existent path raises RuntimeError"""
        with pytest.raises(RuntimeError):
            git_status("/nonexistent/path/that/does/not/exist")

    def test_not_git_repo(self):
        """Test that non-git directory raises RuntimeError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError):
                git_status(tmpdir)


class TestNearestRepo:
    """Tests for nearest_repo function"""

    def test_current_dir_is_repo(self, temp_git_repo):
        """Test when current directory is already the repo root"""
        result = nearest_repo(temp_git_repo)
        assert result == temp_git_repo

    def test_subdir_of_repo(self, temp_git_repo):
        """Test finding repo from a subdirectory"""
        subdir = temp_git_repo / "subdir" / "nested"
        subdir.mkdir(parents=True)

        result = nearest_repo(subdir)
        assert result == temp_git_repo

    def test_deeply_nested_subdir(self, temp_git_repo):
        """Test finding repo from a deeply nested subdirectory"""
        subdir = temp_git_repo / "a" / "b" / "c" / "d"
        subdir.mkdir(parents=True)

        result = nearest_repo(subdir)
        assert result == temp_git_repo

    def test_no_repo_found(self):
        """Test when no repo exists in parent directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            start_dir = Path(tmpdir) / "start"
            start_dir.mkdir()

            result = nearest_repo(start_dir)
            assert result is None

    def test_nested_repos_finds_nearest(self):
        """Test that nested repos find the nearest one"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create outer repo
            outer_repo = root / "outer"
            outer_repo.mkdir()
            subprocess.run(
                ["git", "init"], cwd=outer_repo, capture_output=True, check=True
            )

            # Create inner repo
            inner_repo = outer_repo / "inner"
            inner_repo.mkdir()
            subprocess.run(
                ["git", "init"], cwd=inner_repo, capture_output=True, check=True
            )

            # From inside inner repo, should find inner repo
            subdir = inner_repo / "subdir"
            subdir.mkdir()
            result = nearest_repo(subdir)
            assert result == inner_repo


class TestFarthestRepo:
    """Tests for farthest_repo function"""

    def test_current_dir_is_repo(self, temp_git_repo):
        """Test when current directory is already the repo root"""
        result = farthest_repo(temp_git_repo)
        assert result == temp_git_repo

    def test_subdir_of_repo(self, temp_git_repo):
        """Test finding repo from a subdirectory"""
        subdir = temp_git_repo / "subdir" / "nested"
        subdir.mkdir(parents=True)

        result = farthest_repo(subdir)
        assert result == temp_git_repo

    def test_no_repo_found(self):
        """Test when no repo exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            start_dir = Path(tmpdir) / "start"
            start_dir.mkdir()

            result = farthest_repo(start_dir)
            assert result is None

    def test_nested_repos_finds_farthest(self):
        """Test that nested repos find the farthest (outermost) one"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create outer repo
            outer_repo = root / "outer"
            outer_repo.mkdir()
            subprocess.run(
                ["git", "init"], cwd=outer_repo, capture_output=True, check=True
            )

            # Create inner repo
            inner_repo = outer_repo / "inner"
            inner_repo.mkdir()
            subprocess.run(
                ["git", "init"], cwd=inner_repo, capture_output=True, check=True
            )

            # From inside inner repo subdir, should find outer repo (farthest)
            subdir = inner_repo / "subdir"
            subdir.mkdir()
            result = farthest_repo(subdir)
            assert result == outer_repo

    def test_deeply_nested_repos(self):
        """Test finding farthest repo with deeply nested structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create repos at multiple levels
            level1 = root / "level1"
            level1.mkdir()
            subprocess.run(["git", "init"], cwd=level1, capture_output=True, check=True)

            level2 = level1 / "level2"
            level2.mkdir()
            subprocess.run(["git", "init"], cwd=level2, capture_output=True, check=True)

            level3 = level2 / "level3"
            level3.mkdir()
            subprocess.run(["git", "init"], cwd=level3, capture_output=True, check=True)

            # From level3, should find level1 as farthest
            subdir = level3 / "subdir"
            subdir.mkdir()
            result = farthest_repo(subdir)
            assert result == level1


class TestCurrentCommit:
    """Tests for current_commit function"""

    def test_get_commit_hash(self, temp_git_repo):
        """Test getting commit hash from repo with commits"""
        # Create and commit a file
        (temp_git_repo / "test.txt").write_text("test content")
        subprocess.run(
            ["git", "add", "test.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Test commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        result = current_commit(temp_git_repo)

        # SHA1 should be 40 hex characters
        assert len(result) == 40
        assert all(c in "0123456789abcdef" for c in result)

    def test_no_commits_raises_error(self, temp_git_repo):
        """Test that repo with no commits raises RuntimeError"""
        with pytest.raises(RuntimeError):
            current_commit(temp_git_repo)

    def test_multiple_commits_gets_latest(self, temp_git_repo):
        """Test that we get the latest commit hash"""
        # Create first commit
        (temp_git_repo / "file1.txt").write_text("content1")
        subprocess.run(
            ["git", "add", "file1.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "First commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        first_hash = current_commit(temp_git_repo)

        # Create second commit
        (temp_git_repo / "file2.txt").write_text("content2")
        subprocess.run(
            ["git", "add", "file2.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Second commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        second_hash = current_commit(temp_git_repo)

        # Second hash should be different from first
        assert second_hash != first_hash
        assert len(second_hash) == 40

    def test_subdir_path(self, temp_git_repo):
        """Test getting commit from a subdirectory of the repo"""
        # Create and commit a file
        (temp_git_repo / "test.txt").write_text("test content")
        subprocess.run(
            ["git", "add", "test.txt"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Test commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        # Create a subdirectory and test from there
        subdir = temp_git_repo / "subdir"
        subdir.mkdir()

        result = current_commit(subdir)

        # Should still work from subdirectory
        assert len(result) == 40
        assert all(c in "0123456789abcdef" for c in result)

    def test_nonexistent_path_raises_error(self):
        """Test that non-existent path raises RuntimeError"""
        with pytest.raises(RuntimeError):
            current_commit("/nonexistent/path/that/does/not/exist")

    def test_not_git_repo_raises_error(self):
        """Test that non-git directory raises RuntimeError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError):
                current_commit(tmpdir)
