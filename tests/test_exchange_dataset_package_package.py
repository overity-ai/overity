"""
Unit tests for dataset_package/package.py
"""

import pytest
import tempfile
import tarfile
import json
from pathlib import Path
from overity.exchange.dataset_package.package import (
    package_archive_create,
    metadata_load,
    dataset_load,
)
from overity.exchange import integrity
from overity.model.dataset.metadata import (
    DatasetMetadata,
    DatasetAuthor,
    DatasetMaintainer,
)
from overity.model.dataset.package import DatasetPackageInfo
from overity.errors import MalformedDatasetPackage


class TestDatasetPackagePackage:
    def test_file_sha256(self):
        """Test computing SHA256 hash of a file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()
            path = Path(f.name)

        try:
            result = integrity.file_sha256(path).hexdigest()
            # Known SHA256 hash of "test content"
            assert (
                result
                == "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72"
            )
        finally:
            path.unlink()

    def test_package_archive_create(self):
        """Test creating a dataset package archive."""
        # Create temporary dataset data directory
        with tempfile.TemporaryDirectory() as data_dir:
            data_path = Path(data_dir)

            # Create a test file in the data directory
            test_file = data_path / "test.txt"
            test_file.write_text("test data content")

            # Create metadata
            metadata = DatasetMetadata(
                name="Test Dataset",
                authors=[DatasetAuthor(name="John Doe", email="john@example.com")],
                maintainers=[
                    DatasetMaintainer(name="Jane Smith", email="jane@example.com")
                ],
                description="A test dataset",
            )

            # Create package info
            package_info = DatasetPackageInfo(
                metadata=metadata, dataset_data_path=data_path
            )

            # Create output path
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                output_path = Path(f.name)

            try:
                # Create package archive
                sha256_hash = package_archive_create(package_info, output_path).hexdigest()

                # Verify the archive was created
                assert output_path.exists()
                assert isinstance(sha256_hash, str)

                # Verify the archive contents
                with tarfile.open(output_path, "r:gz") as archive:
                    members = archive.getnames()
                    assert "dataset-metadata.json" in members
                    assert "data" in members
                    assert "data/test.txt" in members

                    # Verify metadata content
                    metadata_file = archive.getmember("dataset-metadata.json")
                    with archive.extractfile(metadata_file) as f:
                        metadata_content = json.load(f)
                        assert metadata_content["name"] == "Test Dataset"

                    # Verify data content
                    data_file = archive.getmember("data/test.txt")
                    with archive.extractfile(data_file) as f:
                        data_content = f.read().decode()
                        assert data_content == "test data content"
            finally:
                output_path.unlink()

    def test_metadata_load(self):
        """Test loading metadata from a dataset package."""
        # Create a temporary package archive
        with tempfile.TemporaryDirectory() as data_dir:
            data_path = Path(data_dir)

            # Create a test file in the data directory
            test_file = data_path / "test.txt"
            test_file.write_text("test data content")

            # Create metadata
            metadata = DatasetMetadata(
                name="Test Dataset",
                authors=[DatasetAuthor(name="John Doe", email="john@example.com")],
                maintainers=[
                    DatasetMaintainer(name="Jane Smith", email="jane@example.com")
                ],
                description="A test dataset",
            )

            # Create package info
            package_info = DatasetPackageInfo(
                metadata=metadata, dataset_data_path=data_path
            )

            # Create output path
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                output_path = Path(f.name)

            try:
                # Create package archive
                package_archive_create(package_info, output_path)

                # Load metadata
                loaded_metadata = metadata_load(output_path)

                # Verify metadata
                assert isinstance(loaded_metadata, DatasetMetadata)
                assert loaded_metadata.name == "Test Dataset"
                assert loaded_metadata.description == "A test dataset"
                assert len(loaded_metadata.authors) == 1
                assert loaded_metadata.authors[0].name == "John Doe"
                assert loaded_metadata.authors[0].email == "john@example.com"
                assert len(loaded_metadata.maintainers) == 1
                assert loaded_metadata.maintainers[0].name == "Jane Smith"
                assert loaded_metadata.maintainers[0].email == "jane@example.com"
            finally:
                output_path.unlink()

    def test_dataset_load(self):
        """Test loading a dataset package and extracting its data."""
        # Create a temporary package archive
        with tempfile.TemporaryDirectory() as data_dir:
            data_path = Path(data_dir)

            # Create a test file in the data directory
            test_file = data_path / "test.txt"
            test_file.write_text("test data content")

            # Create metadata
            metadata = DatasetMetadata(
                name="Test Dataset",
                authors=[DatasetAuthor(name="John Doe", email="john@example.com")],
                maintainers=[
                    DatasetMaintainer(name="Jane Smith", email="jane@example.com")
                ],
                description="A test dataset",
            )

            # Create package info
            package_info = DatasetPackageInfo(
                metadata=metadata, dataset_data_path=data_path
            )

            # Create output path
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                output_path = Path(f.name)

            # Create extraction target directory
            with tempfile.TemporaryDirectory() as target_dir:
                target_path = Path(target_dir)

                try:
                    # Create package archive
                    package_archive_create(package_info, output_path)

                    # Load dataset
                    loaded_metadata = dataset_load(output_path, target_path)

                    # Verify metadata
                    assert isinstance(loaded_metadata, DatasetMetadata)
                    assert loaded_metadata.name == "Test Dataset"

                    # Verify data was extracted
                    extracted_file = target_path / "data" / "test.txt"
                    assert extracted_file.exists()
                    assert extracted_file.read_text() == "test data content"
                finally:
                    output_path.unlink()

    def test_metadata_load_malformed_package(self):
        """Test that loading metadata fails for a malformed package."""
        # Create a temporary archive without metadata
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            archive_path = Path(f.name)

        try:
            # Create an empty archive
            with tarfile.open(archive_path, "w:gz") as archive:
                pass

            # Try to load metadata - should fail
            with pytest.raises(MalformedDatasetPackage):
                metadata_load(archive_path)
        finally:
            archive_path.unlink()
