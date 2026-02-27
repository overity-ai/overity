"""
Unit tests for utils/path.py
"""

import pytest
import tempfile
from pathlib import Path

from overity.utils.path import iter_path, is_subpath


class TestIterPath:
    """Tests for iter_path function"""

    def test_iterates_to_root(self):
        """Test that iter_path iterates from current to root"""
        path = Path("/a/b/c/d")
        paths = list(iter_path(path))

        assert len(paths) == 5
        assert paths[0] == Path("/a/b/c/d")
        assert paths[1] == Path("/a/b/c")
        assert paths[2] == Path("/a/b")
        assert paths[3] == Path("/a")
        assert paths[4] == Path("/")

    def test_single_component(self):
        """Test with single path component"""
        path = Path("/")
        paths = list(iter_path(path))

        assert len(paths) == 1
        assert paths[0] == Path("/")

    def test_relative_path(self):
        """Test with relative path"""
        path = Path("a/b/c")
        paths = list(iter_path(path))

        assert len(paths) == 4
        assert paths[0] == Path("a/b/c")
        assert paths[1] == Path("a/b")
        assert paths[2] == Path("a")
        assert paths[3] == Path(".")


class TestIsSubpath:
    """Tests for is_subpath function with absolute paths"""

    def test_direct_child_absolute(self):
        """Test when a is direct child of b (absolute paths)"""
        a = Path("/home/user/projects/myapp")
        b = Path("/home/user/projects")
        assert is_subpath(a, b) is True

    def test_nested_child_absolute(self):
        """Test when a is nested deeply under b (absolute paths)"""
        a = Path("/home/user/projects/myapp/src/components")
        b = Path("/home/user")
        assert is_subpath(a, b) is True

    def test_same_path_absolute(self):
        """Test when a and b are the same (absolute paths)"""
        a = Path("/home/user/projects")
        b = Path("/home/user/projects")
        assert is_subpath(a, b) is True

    def test_not_subpath_absolute(self):
        """Test when a is not under b (absolute paths)"""
        a = Path("/home/user/projects")
        b = Path("/var/log")
        assert is_subpath(a, b) is False

    def test_sibling_paths_absolute(self):
        """Test sibling paths (absolute)"""
        a = Path("/home/user/projects/myapp")
        b = Path("/home/user/documents")
        assert is_subpath(a, b) is False

    def test_partial_match_absolute(self):
        """Test when path names partially match but aren't related (absolute)"""
        a = Path("/home/user2/projects")
        b = Path("/home/user")
        assert is_subpath(a, b) is False

    def test_child_of_root_absolute(self):
        """Test direct child of root (absolute)"""
        a = Path("/home")
        b = Path("/")
        assert is_subpath(a, b) is True

    def test_root_is_subpath_of_root_absolute(self):
        """Test root is subpath of itself (absolute)"""
        a = Path("/")
        b = Path("/")
        assert is_subpath(a, b) is True


class TestIsSubpathRelative:
    """Tests for is_subpath function with relative paths"""

    def test_direct_child_relative(self):
        """Test when a is direct child of b (relative paths)"""
        a = Path("projects/myapp")
        b = Path("projects")
        assert is_subpath(a, b) is True

    def test_nested_child_relative(self):
        """Test when a is nested deeply under b (relative paths)"""
        a = Path("src/components/buttons")
        b = Path("src")
        assert is_subpath(a, b) is True

    def test_same_path_relative(self):
        """Test when a and b are the same (relative paths)"""
        a = Path("projects/myapp")
        b = Path("projects/myapp")
        assert is_subpath(a, b) is True

    def test_not_subpath_relative(self):
        """Test when a is not under b (relative paths)"""
        a = Path("projects/myapp")
        b = Path("documents/reports")
        assert is_subpath(a, b) is False

    def test_sibling_paths_relative(self):
        """Test sibling paths (relative)"""
        a = Path("src/components")
        b = Path("tests/unit")
        assert is_subpath(a, b) is False

    def test_going_up_relative(self):
        """Test when a goes up from b (relative paths)"""
        a = Path("..")
        b = Path("src/components")
        assert is_subpath(a, b) is False

    def test_single_component_relative(self):
        """Test single component relative paths"""
        a = Path("src")
        b = Path(".")
        assert is_subpath(a, b) is True

    def test_dot_current_directory(self):
        """Test with current directory"""
        a = Path("./src/components")
        b = Path(".")
        assert is_subpath(a, b) is True


class TestIsSubpathMixed:
    """Tests for is_subpath function with mixed absolute/relative paths"""

    def test_absolute_a_relative_b(self):
        """Test absolute a with relative b"""
        a = Path("/home/user/projects")
        b = Path("home/user")
        # This will fail because absolute vs relative don't mix well
        assert is_subpath(a, b) is False

    def test_relative_a_absolute_b(self):
        """Test relative a with absolute b"""
        a = Path("projects/myapp")
        b = Path("/home/user")
        assert is_subpath(a, b) is False


class TestIsSubpathWithRealFilesystem:
    """Tests for is_subpath with actual filesystem paths"""

    def test_real_subpath(self, tmp_path):
        """Test with real directory structure"""
        # Create nested directories
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)

        assert is_subpath(nested, tmp_path) is True
        assert is_subpath(nested, tmp_path / "a") is True
        assert is_subpath(nested, tmp_path / "a" / "b") is True

    def test_real_not_subpath(self, tmp_path):
        """Test paths that exist but aren't related"""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        assert is_subpath(dir1, dir2) is False
        assert is_subpath(dir2, dir1) is False

    def test_real_file_path(self, tmp_path):
        """Test with file paths"""
        nested_dir = tmp_path / "src" / "utils"
        nested_dir.mkdir(parents=True)
        file_path = nested_dir / "helpers.py"
        file_path.write_text("# helper functions")

        assert is_subpath(file_path, tmp_path) is True
        assert is_subpath(file_path, nested_dir) is True
        assert is_subpath(file_path, tmp_path / "tests") is False
