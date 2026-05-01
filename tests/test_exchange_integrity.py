"""Unit tests for the overity.exchange.integrity module."""

import hashlib
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, mock_open

from overity.exchange.integrity import file_sha256


class TestFileSHA256:
    """Test cases for the file_sha256 function."""
    
    def test_file_sha256_with_known_content(self):
        """Test SHA256 computation with known content."""
        # Create a temporary file with known content
        test_content = b"Hello, World!"
        expected_hash = hashlib.sha256(test_content).hexdigest()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = Path(temp_file.name)
        
        try:
            # Compute hash using our function
            result = file_sha256(temp_file_path)
            assert result.hexdigest() == expected_hash
        finally:
            # Clean up
            temp_file_path.unlink()
    
    def test_file_sha256_empty_file(self):
        """Test SHA256 computation with empty file."""
        # Expected hash for empty content
        expected_hash = hashlib.sha256(b"").hexdigest()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = Path(temp_file.name)
        
        try:
            result = file_sha256(temp_file_path)
            assert result.hexdigest() == expected_hash
        finally:
            temp_file_path.unlink()
    
    def test_file_sha256_large_file(self):
        """Test SHA256 computation with larger content."""
        # Create content that's larger than typical buffer sizes
        test_content = b"A" * 1024 * 1024  # 1MB of 'A's
        expected_hash = hashlib.sha256(test_content).hexdigest()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = Path(temp_file.name)
        
        try:
            result = file_sha256(temp_file_path)
            assert result.hexdigest() == expected_hash
        finally:
            temp_file_path.unlink()
    
    def test_file_sha256_binary_content(self):
        """Test SHA256 computation with binary content."""
        # Create binary content with various byte values
        test_content = bytes(range(256))  # All possible byte values 0-255
        expected_hash = hashlib.sha256(test_content).hexdigest()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = Path(temp_file.name)
        
        try:
            result = file_sha256(temp_file_path)
            assert result.hexdigest() == expected_hash
        finally:
            temp_file_path.unlink()
    
    def test_file_sha256_nonexistent_file(self):
        """Test error handling for non-existent file."""
        nonexistent_path = Path("/tmp/nonexistent_file_12345.txt")
        
        with pytest.raises(FileNotFoundError):
            file_sha256(nonexistent_path)
    
    def test_file_sha256_directory_path(self):
        """Test error handling when given a directory path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            with pytest.raises(IsADirectoryError):
                file_sha256(dir_path)
    
    def test_file_sha256_string_path(self):
        """Test that function accepts string paths in addition to Path objects."""
        test_content = b"Test content for string path"
        expected_hash = hashlib.sha256(test_content).hexdigest()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name  # String path
        
        try:
            result = file_sha256(temp_file_path)
            assert result.hexdigest() == expected_hash
        finally:
            Path(temp_file_path).unlink()
    
    def test_file_sha256_result_has_hexdigest_method(self):
        """Test that the returned object has the expected hexdigest method."""
        test_content = b"Test content"
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = Path(temp_file.name)
        
        try:
            result = file_sha256(temp_file_path)
            # Should have hexdigest method
            assert hasattr(result, 'hexdigest')
            # Should return a string
            hex_digest = result.hexdigest()
            assert isinstance(hex_digest, str)
            # Should be 64 characters (256 bits = 64 hex chars)
            assert len(hex_digest) == 64
            # Should be lowercase hex
            assert hex_digest.islower()
            # Should only contain hex characters
            assert all(c in '0123456789abcdef' for c in hex_digest)
        finally:
            temp_file_path.unlink()
    
    def test_file_sha256_consistency(self):
        """Test that the same file produces the same hash consistently."""
        test_content = b"Consistent content test"
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = Path(temp_file.name)
        
        try:
            # Compute hash multiple times
            hash1 = file_sha256(temp_file_path).hexdigest()
            hash2 = file_sha256(temp_file_path).hexdigest()
            hash3 = file_sha256(temp_file_path).hexdigest()
            
            # All should be identical
            assert hash1 == hash2 == hash3
        finally:
            temp_file_path.unlink()
    
    def test_file_sha256_different_files_different_hashes(self):
        """Test that different files produce different hashes."""
        content1 = b"Content number one"
        content2 = b"Content number two"
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file1:
            temp_file1.write(content1)
            temp_file1_path = Path(temp_file1.name)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file2:
            temp_file2.write(content2)
            temp_file2_path = Path(temp_file2.name)
        
        try:
            hash1 = file_sha256(temp_file1_path).hexdigest()
            hash2 = file_sha256(temp_file2_path).hexdigest()
            
            # Different content should produce different hashes
            assert hash1 != hash2
        finally:
            temp_file1_path.unlink()
            temp_file2_path.unlink()
    
    def test_file_sha256_permission_error(self):
        """Test handling of permission errors."""
        test_content = b"Permission test content"
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = Path(temp_file.name)
        
        try:
            # Mock the open function to raise PermissionError
            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                with pytest.raises(PermissionError):
                    file_sha256(temp_file_path)
        finally:
            temp_file_path.unlink()
    
    def test_file_sha256_unicode_content(self):
        """Test SHA256 computation with unicode content."""
        # UTF-8 encoded unicode content
        unicode_text = "Hello 世界! 🌍 Ñoël"
        test_content = unicode_text.encode('utf-8')
        expected_hash = hashlib.sha256(test_content).hexdigest()
        
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            temp_file.write(unicode_text)
            temp_file_path = Path(temp_file.name)
        
        try:
            result = file_sha256(temp_file_path)
            assert result.hexdigest() == expected_hash
        finally:
            temp_file_path.unlink()
    
    def test_file_sha256_special_chars_in_filename(self):
        """Test with filenames containing special characters."""
        test_content = b"Special filename test content"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file with special characters in name
            special_filename = "test_file-with_special.chars!@#$%.txt"
            file_path = Path(temp_dir) / special_filename
            
            file_path.write_bytes(test_content)
            
            expected_hash = hashlib.sha256(test_content).hexdigest()
            result = file_sha256(file_path)
            assert result.hexdigest() == expected_hash


class TestFileSHA256Integration:
    """Integration tests for file_sha256 with package-related scenarios."""
    
    def test_file_sha256_package_like_scenario(self):
        """Test hash computation in a scenario similar to package creation."""
        # Simulate content that might be in a package
        package_content = {
            'model.json': b'{"name": "test_model", "version": "1.0"}',
            'weights.bin': b'\x00\x01\x02\x03' * 1000,  # Binary weights
            'metadata.yaml': b'name: test_model\nversion: 1.0\nframework: pytorch\n',
        }
        
        computed_hashes = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files and compute hashes
            for filename, content in package_content.items():
                file_path = temp_path / filename
                file_path.write_bytes(content)
                
                # Compute hash using our function
                result = file_sha256(file_path)
                computed_hashes[filename] = result.hexdigest()
                
                # Verify against expected hash
                expected_hash = hashlib.sha256(content).hexdigest()
                assert computed_hashes[filename] == expected_hash
        
        # Verify all hashes are different (different content)
        hash_values = list(computed_hashes.values())
        assert len(set(hash_values)) == len(hash_values), "All files should have unique hashes"
    
    def test_file_sha256_large_binary_file(self):
        """Test with a large binary file similar to model weights."""
        # Create 5MB of pseudo-random binary data
        import os
        large_content = os.urandom(5 * 1024 * 1024)
        expected_hash = hashlib.sha256(large_content).hexdigest()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(large_content)
            temp_file_path = Path(temp_file.name)
        
        try:
            result = file_sha256(temp_file_path)
            assert result.hexdigest() == expected_hash
        finally:
            temp_file_path.unlink()