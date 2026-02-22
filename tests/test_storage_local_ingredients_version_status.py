"""
Unit tests for ingredients_version_status function
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from overity.storage.local import LocalStorage
from overity.model.versioning import VersioningStatus


class TestIngredientsVersionStatus:
    """Tests for ingredients_version_status function"""

    def test_not_versioned_no_git_repo(self):
        """Test when there is no git repository"""
        with patch("overity.storage.local.git_utils.nearest_repo") as mock_nearest:
            mock_nearest.return_value = None

            storage = LocalStorage(Path("/test/program"))
            result = storage.ingredients_version_status()

            assert result == VersioningStatus.NotVersioned
            mock_nearest.assert_called_once_with(storage.ingredients_folder)

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
            mock_nearest.assert_called_once_with(storage.ingredients_folder)
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

    def test_dirty_multiple_changes_some_inside(self):
        """Test when some changes are inside and some outside ingredients"""
        with patch(
            "overity.storage.local.git_utils.nearest_repo"
        ) as mock_nearest, patch(
            "overity.storage.local.git_utils.git_status"
        ) as mock_git_status, patch(
            "overity.storage.local.path_utils.is_subpath"
        ) as mock_is_subpath:
            mock_nearest.return_value = Path("/test/program")

            # Create mock changes - one inside, one outside
            mock_change1 = MagicMock()
            mock_change1.from_path = "other_folder/file.txt"
            mock_change2 = MagicMock()
            mock_change2.from_path = "ingredients/measurement_qualification/test.py"
            mock_git_status.return_value = [mock_change1, mock_change2]

            # First call (outside) returns False, second call (inside) returns True
            mock_is_subpath.side_effect = [False, True]

            storage = LocalStorage(Path("/test/program"))
            result = storage.ingredients_version_status()

            assert result == VersioningStatus.Dirty
            assert mock_is_subpath.call_count == 2

    def test_ingredients_at_repo_root(self):
        """Test when ingredients folder is at the repo root"""
        with patch(
            "overity.storage.local.git_utils.nearest_repo"
        ) as mock_nearest, patch(
            "overity.storage.local.git_utils.git_status"
        ) as mock_git_status, patch(
            "overity.storage.local.path_utils.is_subpath"
        ) as mock_is_subpath:
            # Ingredients folder IS the repo root
            mock_nearest.return_value = Path("/test/program/ingredients")

            # Some changes in the repo
            mock_change = MagicMock()
            mock_change.from_path = "training_optimization/new_method.py"
            mock_git_status.return_value = [mock_change]

            # All changes are within ingredients (since ingredients is the root)
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

    def test_repo_in_parent_directory(self):
        """Test when repo is in a parent directory of ingredients"""
        with patch(
            "overity.storage.local.git_utils.nearest_repo"
        ) as mock_nearest, patch(
            "overity.storage.local.git_utils.git_status"
        ) as mock_git_status, patch(
            "overity.storage.local.path_utils.is_subpath"
        ) as mock_is_subpath:
            # Repo is two levels up from ingredients
            mock_nearest.return_value = Path("/test")

            # Changes in various locations
            mock_change = MagicMock()
            mock_change.from_path = "program/ingredients/lib/helper.py"
            mock_git_status.return_value = [mock_change]

            mock_is_subpath.return_value = True

            storage = LocalStorage(Path("/test/program"))
            result = storage.ingredients_version_status()

            assert result == VersioningStatus.Dirty
            # Verify git_status was called with the repo path
            mock_git_status.assert_called_once_with("/test")

    def test_empty_git_status_clean(self):
        """Test that empty git status results in Clean"""
        with patch(
            "overity.storage.local.git_utils.nearest_repo"
        ) as mock_nearest, patch(
            "overity.storage.local.git_utils.git_status"
        ) as mock_git_status:
            mock_nearest.return_value = Path("/test/program")
            mock_git_status.return_value = []

            storage = LocalStorage(Path("/test/program"))
            result = storage.ingredients_version_status()

            assert result == VersioningStatus.Clean

    def test_all_changes_filtered_out_clean(self):
        """Test when all changes are outside ingredients folder"""
        with patch(
            "overity.storage.local.git_utils.nearest_repo"
        ) as mock_nearest, patch(
            "overity.storage.local.git_utils.git_status"
        ) as mock_git_status, patch(
            "overity.storage.local.path_utils.is_subpath"
        ) as mock_is_subpath:
            mock_nearest.return_value = Path("/test/program")

            # Multiple changes but all outside ingredients
            mock_changes = [
                MagicMock(from_path="shelf/report.json"),
                MagicMock(from_path="catalyst/bench.toml"),
                MagicMock(from_path="README.md"),
            ]
            mock_git_status.return_value = mock_changes

            # All changes are outside ingredients
            mock_is_subpath.return_value = False

            storage = LocalStorage(Path("/test/program"))
            result = storage.ingredients_version_status()

            assert result == VersioningStatus.Clean
            assert mock_is_subpath.call_count == 3
