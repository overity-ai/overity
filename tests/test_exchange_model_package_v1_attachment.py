"""
Unit tests for model_package_v1/attachment.py
"""

import pytest
import tempfile
import json
import tarfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from overity.exchange.model_package_v1.attachment import (
    meta_decode,
    meta_encode,
    meta_from_file,
    integrity_check,
)
from overity.model.ml_model.attachment import AttachmentMetadata, ExtractedAttachment
from overity.errors import AttachmentIntegrityError


class TestAttachmentDecode:
    """Test the meta_decode function"""

    def test_valid_meta_decode(self):
        """Test decoding valid attachment data"""
        data = {
            "filename": "test_file.pdf",
            "sha256_hash": "abc123def456",
            "mimetype": "application/pdf",
            "description": "Test PDF file"
        }
        
        result =             meta_decode(data)
        
        assert isinstance(result, AttachmentMetadata)
        assert result.filename == "test_file.pdf"
        assert result.sha256_hash == "abc123def456"
        assert result.mimetype == "application/pdf"
        assert result.description == "Test PDF file"

    def test_valid_meta_decode_without_description(self):
        """Test decoding valid attachment data without description"""
        data = {
            "filename": "test_file.txt",
            "sha256_hash": "def456abc123",
            "mimetype": "text/plain"
        }
        
        result =             meta_decode(data)
        
        assert isinstance(result, AttachmentMetadata)
        assert result.filename == "test_file.txt"
        assert result.sha256_hash == "def456abc123"
        assert result.mimetype == "text/plain"
        assert result.description is None

    def test_meta_decode_missing_required_field(self):
        """Test decoding attachment data missing required fields"""
        data = {
            "filename": "test_file.txt",
            # Missing sha256_hash
            "mimetype": "text/plain"
        }
        
        with pytest.raises(KeyError):
            meta_decode(data)

    def test_meta_decode_empty_data(self):
        """Test decoding empty attachment data"""
        data = {}
        
        with pytest.raises(KeyError):
            meta_decode(data)


class TestAttachmentEncode:
    """Test the meta_encode function"""

    def test_valid_meta_encode_with_description(self):
        """Test encoding attachment metadata with description"""
        attachment = AttachmentMetadata(
            filename="test_file.pdf",
            sha256_hash="abc123def456",
            mimetype="application/pdf",
            description="Test PDF file"
        )
        
        result = meta_encode(attachment)
        
        assert result == {
            "filename": "test_file.pdf",
            "sha256_hash": "abc123def456",
            "mimetype": "application/pdf",
            "description": "Test PDF file"
        }

    def test_valid_meta_encode_without_description(self):
        """Test encoding attachment metadata without description"""
        attachment = AttachmentMetadata(
            filename="test_file.txt",
            sha256_hash="def456abc123",
            mimetype="text/plain"
        )
        
        result = meta_encode(attachment)
        
        assert result == {
            "filename": "test_file.txt",
            "sha256_hash": "def456abc123",
            "mimetype": "text/plain"
        }

    def test_meta_encode_with_empty_description(self):
        """Test encoding attachment metadata with empty description"""
        attachment = AttachmentMetadata(
            filename="test_file.txt",
            sha256_hash="def456abc123",
            mimetype="text/plain",
            description=""
        )
        
        result = meta_encode(attachment)
        
        assert result == {
            "filename": "test_file.txt",
            "sha256_hash": "def456abc123",
            "mimetype": "text/plain"
        }


class TestAttachmentMetaFromFile:
    """Test the meta_from_file function"""

    def test_attachment_meta_from_text_file(self):
        """Test creating attachment metadata from a text file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is test content for attachment")
            f.flush()
            file_path = Path(f.name)
        
        try:
            result = meta_from_file(file_path, description="Test text file")
            
            assert isinstance(result, AttachmentMetadata)
            assert result.filename == file_path.name
            assert result.mimetype == "text/plain"
            assert result.description == "Test text file"
            assert len(result.sha256_hash) == 64  # SHA256 hex digest length
            
            # Verify the hash is a valid hex string (64 characters for SHA256)
            assert len(result.sha256_hash) == 64
            assert all(c in '0123456789abcdef' for c in result.sha256_hash)
            
        finally:
            file_path.unlink()

    def test_attachment_meta_from_pdf_file(self):
        """Test creating attachment metadata from a PDF file"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b"%PDF-1.4 fake pdf content")
            f.flush()
            file_path = Path(f.name)
        
        try:
            result = meta_from_file(file_path)
            
            assert isinstance(result, AttachmentMetadata)
            assert result.filename == file_path.name
            assert result.mimetype == "application/pdf"
            assert result.description is None
            assert len(result.sha256_hash) == 64
            
        finally:
            file_path.unlink()

    def test_attachment_meta_from_unknown_file_type(self):
        """Test creating attachment metadata from a file with unknown extension"""
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as f:
            f.write(b"some unknown file content")
            f.flush()
            file_path = Path(f.name)
        
        try:
            result = meta_from_file(file_path, description="Unknown file")
            
            assert isinstance(result, AttachmentMetadata)
            assert result.filename == file_path.name
            assert result.mimetype is None  # Unknown file types return None
            assert result.description == "Unknown file"
            assert len(result.sha256_hash) == 64
            
        finally:
            file_path.unlink()

    def test_attachment_meta_from_binary_file(self):
        """Test creating attachment metadata from a binary file"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF')  # JPEG header
            f.write(b'more fake image data')
            f.flush()
            file_path = Path(f.name)
        
        try:
            result = meta_from_file(file_path)
            
            assert isinstance(result, AttachmentMetadata)
            assert result.filename == file_path.name
            assert result.mimetype == "image/jpeg"
            assert result.description is None
            assert len(result.sha256_hash) == 64
            
        finally:
            file_path.unlink()

    def test_attachment_meta_from_nonexistent_file(self):
        """Test creating attachment metadata from a non-existent file"""
        nonexistent_path = Path("/path/that/does/not/exist.txt")
        
        with pytest.raises(FileNotFoundError):
            meta_from_file(nonexistent_path)

    @patch('overity.exchange.model_package_v1.attachment.integrity.file_sha256')
    def test_meta_from_file_integrity_error(self, mock_file_sha256):
        """Test handling of integrity.file_sha256 errors"""
        mock_file_sha256.side_effect = Exception("Integrity error")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            f.flush()
            file_path = Path(f.name)
        
        try:
            with pytest.raises(Exception):
                meta_from_file(file_path)
        finally:
            file_path.unlink()


class TestAttachmentIntegrityCheck:
    """Test the integrity_check function"""

    def test_valid_integrity_check(self):
        """Test successful attachment integrity check"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for integrity check")
            f.flush()
            file_path = Path(f.name)
        
        try:
            # Create metadata with correct hash
            import hashlib
            with open(file_path, 'rb') as file:
                digest = hashlib.file_digest(file, 'sha256')
                correct_hash = digest.hexdigest()
            
            meta = AttachmentMetadata(
                filename="test_file",
                sha256_hash=correct_hash,
                mimetype="text/plain"
            )
            
            # Should not raise any exception
            integrity_check(file_path, meta)
            
        finally:
            file_path.unlink()

    def test_invalid_integrity_check(self):
        """Test attachment integrity check with invalid hash"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for integrity check")
            f.flush()
            file_path = Path(f.name)
        
        try:
            # Create metadata with incorrect hash
            meta = AttachmentMetadata(
                filename="test_file",
                sha256_hash="incorrect_hash_value_123456789012345678901234567890123456789012345678901234567890",
                mimetype="text/plain"
            )
            
            with pytest.raises(AttachmentIntegrityError) as exc_info:
                integrity_check(file_path, meta)
            
            # Verify the error message contains expected information
            error_msg = str(exc_info.value)
            assert "Attachment integrity error" in error_msg
            assert str(file_path) in error_msg
            assert "incorrect_hash_value" in error_msg
            
        finally:
            file_path.unlink()

    def test_integrity_check_nonexistent_file(self):
        """Test attachment integrity check with non-existent file"""
        nonexistent_path = Path("/path/that/does/not/exist.txt")
        
        meta = AttachmentMetadata(
            filename="test_file",
            sha256_hash="some_hash_value",
            mimetype="text/plain"
        )
        
        with pytest.raises(FileNotFoundError):
                integrity_check(nonexistent_path, meta)

    @patch('overity.exchange.model_package_v1.attachment.integrity.file_sha256')
    def test_integrity_check_integrity_error(self, mock_file_sha256):
        """Test handling of integrity.file_sha256 errors during integrity check"""
        mock_file_sha256.side_effect = Exception("Integrity error")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            f.flush()
            file_path = Path(f.name)
        
        meta = AttachmentMetadata(
            filename="test_file",
            sha256_hash="some_hash_value",
            mimetype="text/plain"
        )
        
        try:
            with pytest.raises(Exception):
                integrity_check(file_path, meta)
        finally:
            file_path.unlink()


class TestAttachmentRoundTrip:
    """Test encoding and decoding round trips"""

    def test_attachment_round_trip_with_description(self):
        """Test encoding then decoding attachment metadata with description"""
        original = AttachmentMetadata(
            filename="test_document.pdf",
            sha256_hash="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            mimetype="application/pdf",
            description="Important test document"
        )
        
        # Encode then decode
        encoded = meta_encode(original)
        decoded = meta_decode(encoded)
        
        # Verify round trip preserves all data
        assert decoded.filename == original.filename
        assert decoded.sha256_hash == original.sha256_hash
        assert decoded.mimetype == original.mimetype
        assert decoded.description == original.description

    def test_attachment_round_trip_without_description(self):
        """Test encoding then decoding attachment metadata without description."""
        original = AttachmentMetadata(
            filename="simple_file.txt",
            sha256_hash="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            mimetype="text/plain"
        )
        
        # Encode then decode
        encoded = meta_encode(original)
        decoded = meta_decode(encoded)
        
        # Verify round trip preserves all data
        assert decoded.filename == original.filename
        assert decoded.sha256_hash == original.sha256_hash
        assert decoded.mimetype == original.mimetype
        assert decoded.description is None

    def test_meta_encode_with_none_mimetype(self):
        """Test encoding attachment metadata with mimetype=None."""
        attachment = AttachmentMetadata(
            filename="unknown_file.xyz",
            sha256_hash="abc123def4567890abc123def4567890abc123def4567890abc123def4567890",
            mimetype=None
        )
        
        result = meta_encode(attachment)
        
        # Should not include mimetype field when it's None
        assert result == {
            "filename": "unknown_file.xyz",
            "sha256_hash": "abc123def4567890abc123def4567890abc123def4567890abc123def4567890"
        }
        assert "mimetype" not in result

    def test_meta_encode_with_none_mimetype_and_description(self):
        """Test encoding attachment metadata with both mimetype=None and no description."""
        attachment = AttachmentMetadata(
            filename="minimal_file.bin",
            sha256_hash="def456abc1237890def456abc1237890def456abc1237890def456abc1237890",
            mimetype=None,
            description=None
        )
        
        result = meta_encode(attachment)
        
        # Should only include required fields
        assert result == {
            "filename": "minimal_file.bin",
            "sha256_hash": "def456abc1237890def456abc1237890def456abc1237890def456abc1237890"
        }
        assert "mimetype" not in result
        assert "description" not in result

    def test_meta_decode_without_mimetype_field(self):
        """Test decoding attachment data when mimetype field is missing."""
        data = {
            "filename": "file_without_mimetype.txt",
            "sha256_hash": "abc123def4567890abc123def4567890abc123def4567890abc123def4567890"
        }
        
        result = meta_decode(data)
        
        assert isinstance(result, AttachmentMetadata)
        assert result.filename == "file_without_mimetype.txt"
        assert result.sha256_hash == "abc123def4567890abc123def4567890abc123def4567890abc123def4567890"
        assert result.mimetype is None
        assert result.description is None

    def test_meta_decode_with_mimetype_none(self):
        """Test decoding attachment data when mimetype is explicitly None."""
        data = {
            "filename": "file_with_none_mimetype.txt",
            "sha256_hash": "def456abc1237890def456abc1237890def456abc1237890def456abc1237890",
            "mimetype": None
        }
        
        result = meta_decode(data)
        
        assert isinstance(result, AttachmentMetadata)
        assert result.filename == "file_with_none_mimetype.txt"
        assert result.sha256_hash == "def456abc1237890def456abc1237890def456abc1237890def456abc1237890"
        assert result.mimetype is None
        assert result.description is None

    def test_meta_decode_with_missing_mimetype_but_with_description(self):
        """Test decoding attachment data when mimetype is missing but description is present."""
        data = {
            "filename": "file_with_description_only.md",
            "sha256_hash": "xyz789uvw456abc123def4567890abc123def4567890abc123def4567890abc",
            "description": "File with description but no mimetype"
        }
        
        result = meta_decode(data)
        
        assert isinstance(result, AttachmentMetadata)
        assert result.filename == "file_with_description_only.md"
        assert result.sha256_hash == "xyz789uvw456abc123def4567890abc123def4567890abc123def4567890abc"
        assert result.mimetype is None
        assert result.description == "File with description but no mimetype"

    def test_round_trip_with_none_mimetype(self):
        """Test round-trip encoding and decoding with mimetype=None."""
        original = AttachmentMetadata(
            filename="unknown_type.xyz",
            sha256_hash="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            mimetype=None,
            description="File with unknown type"
        )
        
        # Encode then decode
        encoded = meta_encode(original)
        decoded = meta_decode(encoded)
        
        # Verify round trip preserves all data
        assert decoded.filename == original.filename
        assert decoded.sha256_hash == original.sha256_hash
        assert decoded.mimetype is None  # Should remain None
        assert decoded.description == original.description
        
        # Verify encoded object doesn't include mimetype field
        assert "mimetype" not in encoded
        assert encoded["description"] == "File with unknown type"

    def test_round_trip_with_none_mimetype_no_description(self):
        """Test round-trip encoding and decoding with mimetype=None and no description."""
        original = AttachmentMetadata(
            filename="minimal_file.tmp",
            sha256_hash="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            mimetype=None,
            description=None
        )
        
        # Encode then decode
        encoded = meta_encode(original)
        decoded = meta_decode(encoded)
        
        # Verify round trip preserves all data
        assert decoded.filename == original.filename
        assert decoded.sha256_hash == original.sha256_hash
        assert decoded.mimetype is None
        assert decoded.description is None
        
        # Verify encoded object only includes required fields
        assert encoded == {
            "filename": "minimal_file.tmp",
            "sha256_hash": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        }
        assert "mimetype" not in encoded
        assert "description" not in encoded

    def test_process_attachment_missing_file(self):
        """Test _process_attachment when attachment file is missing from archive."""
        from overity.exchange.model_package_v1.package import _process_attachment
        from overity.errors import MalformedModelPackage
        
        # Create a fake tarfile with no attachment
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            archive_path = Path(f.name)
        
        # Create empty archive
        with tarfile.open(archive_path, "w:gz") as archive:
            pass  # Empty archive
        
        # Create target folder
        with tempfile.TemporaryDirectory() as target_dir:
            target_path = Path(target_dir)
            
            # Create attachment metadata
            att_meta = AttachmentMetadata(
                filename="missing.txt",
                sha256_hash="abc123def4567890abc123def4567890abc123def4567890abc123def4567890",
                mimetype="text/plain"
            )
            
            try:
                # Open archive for reading
                with tarfile.open(archive_path, "r:gz") as archive:
                    # Should raise MalformedModelPackage
                    with pytest.raises(MalformedModelPackage) as exc_info:
                        _process_attachment(archive_path, archive, target_path, att_meta)
                    
                    # Verify error message
                    assert "missing.txt" in str(exc_info.value)
                    assert "not found in archive" in str(exc_info.value)
            finally:
                archive_path.unlink()

    def test_process_attachment_success(self):
        """Test _process_attachment when attachment file exists in archive."""
        from overity.exchange.model_package_v1.package import _process_attachment
        
        # Create attachment content
        att_content = b"This is attachment content for testing"
        
        # Create temporary file with attachment content
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(att_content)
            att_file_path = Path(f.name)
        
        # Create archive with attachment
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            archive_path = Path(f.name)
        
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(str(att_file_path), arcname="attachments/test_file.txt")
        
        # Create target folder
        with tempfile.TemporaryDirectory() as target_dir:
            target_path = Path(target_dir)
            
            # Create attachment metadata
            att_meta = AttachmentMetadata(
                filename="test_file.txt",
                sha256_hash="dummy_hash",
                mimetype="text/plain",
                description="Test attachment"
            )
            
            try:
                # Open archive for reading
                with tarfile.open(archive_path, "r:gz") as archive:
                    # Process attachment
                    result = _process_attachment(archive_path, archive, target_path, att_meta)
                    
                    # Verify result
                    assert isinstance(result, ExtractedAttachment)
                    assert result.meta.filename == "test_file.txt"
                    assert result.meta.mimetype == "text/plain"
                    assert result.meta.description == "Test attachment"
                    assert result.path.exists()
                    assert result.path.read_bytes() == att_content
                    
                    # Verify file was extracted to correct location
                    expected_path = target_path / "test_file.txt"
                    assert result.path == expected_path
                    assert expected_path.exists()
            finally:
                archive_path.unlink()
                att_file_path.unlink()
