"""
Unit tests for inference_agent_package/package.py
"""

import pytest
import tempfile
import tarfile
import json
from pathlib import Path
from overity.exchange.inference_agent_package.package import (
    package_archive_create,
    metadata_load,
    agent_load,
)
from overity.exchange import integrity
from overity.model.inference_agent.metadata import (
    InferenceAgentMetadata,
    InferenceAgentAuthor,
    InferenceAgentMaintainer,
)
from overity.model.inference_agent.package import InferenceAgentPackageInfo
from overity.errors import MalformedAgentPackage


class TestInferenceAgentPackagePackage:
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
        """Test creating an inference agent package archive."""
        # Create temporary agent data directory
        with tempfile.TemporaryDirectory() as data_dir:
            data_path = Path(data_dir)

            # Create test files in the data directory
            agent_file = data_path / "agent.py"
            agent_file.write_text(
                "# Inference agent implementation\nprint('Hello Agent')"
            )

            config_file = data_path / "config.json"
            config_file.write_text('{"param": "value"}')

            # Create metadata
            metadata = InferenceAgentMetadata(
                name="Test Inference Agent",
                version="1.0.0",
                authors=[
                    InferenceAgentAuthor(name="John Doe", email="john@example.com")
                ],
                maintainers=[
                    InferenceAgentMaintainer(
                        name="Jane Smith", email="jane@example.com"
                    )
                ],
                capabilities=frozenset(["text-generation", "summarization"]),
                compatible_targets=frozenset(["cpu", "gpu"]),
                compatible_tags=frozenset(["latest", "stable"]),
            )

            # Create package info
            package_info = InferenceAgentPackageInfo(
                metadata=metadata,
                agent_data_path=data_path,
            )

            # Create output path
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                output_path = Path(f.name)

            try:
                # Create package archive
                sha256_hash = package_archive_create(
                    package_info, output_path
                ).hexdigest()

                # Verify the archive was created
                assert output_path.exists()
                assert isinstance(sha256_hash, str)

                # Verify the archive contents
                with tarfile.open(output_path, "r:gz") as archive:
                    members = archive.getnames()
                    assert "agent-metadata.json" in members
                    assert "data" in members
                    assert "data/agent.py" in members
                    assert "data/config.json" in members

                    # Verify metadata content
                    metadata_file = archive.getmember("agent-metadata.json")
                    f = archive.extractfile(metadata_file)
                    assert f is not None
                    with f:
                        metadata_content = json.load(f)
                        assert metadata_content["name"] == "Test Inference Agent"

                    # Verify agent data content
                    agent_py_file = archive.getmember("data/agent.py")
                    f = archive.extractfile(agent_py_file)
                    assert f is not None
                    with f:
                        agent_content = f.read()
                        assert (
                            agent_content
                            == b"# Inference agent implementation\nprint('Hello Agent')"
                        )

                    # Verify config content
                    config_json_file = archive.getmember("data/config.json")
                    f = archive.extractfile(config_json_file)
                    assert f is not None
                    with f:
                        config_content = f.read()
                        assert config_content == b'{"param": "value"}'
            finally:
                output_path.unlink()

    def test_metadata_load(self):
        """Test loading metadata from an inference agent package."""
        # Create temporary agent data directory
        with tempfile.TemporaryDirectory() as data_dir:
            data_path = Path(data_dir)

            # Create test files in the data directory
            agent_file = data_path / "agent.py"
            agent_file.write_text("# Inference agent implementation")

            # Create metadata
            metadata = InferenceAgentMetadata(
                name="Test Inference Agent",
                version="1.0.0",
                authors=[
                    InferenceAgentAuthor(name="John Doe", email="john@example.com")
                ],
                maintainers=[
                    InferenceAgentMaintainer(
                        name="Jane Smith", email="jane@example.com"
                    )
                ],
                capabilities=frozenset(["text-generation", "summarization"]),
                compatible_targets=frozenset(["cpu", "gpu"]),
                compatible_tags=frozenset(["latest", "stable"]),
            )

            # Create package info
            package_info = InferenceAgentPackageInfo(
                metadata=metadata,
                agent_data_path=data_path,
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
                assert isinstance(loaded_metadata, InferenceAgentMetadata)
                assert loaded_metadata.name == "Test Inference Agent"
                assert loaded_metadata.version == "1.0.0"
                assert loaded_metadata.capabilities == frozenset(
                    ["text-generation", "summarization"]
                )
                assert loaded_metadata.compatible_targets == frozenset(["cpu", "gpu"])
                assert loaded_metadata.compatible_tags == frozenset(
                    ["latest", "stable"]
                )
                assert len(loaded_metadata.authors) == 1
                assert loaded_metadata.authors[0].name == "John Doe"
                assert loaded_metadata.authors[0].email == "john@example.com"
                assert len(loaded_metadata.maintainers) == 1
                assert loaded_metadata.maintainers[0].name == "Jane Smith"
                assert loaded_metadata.maintainers[0].email == "jane@example.com"
            finally:
                output_path.unlink()

    def test_agent_load(self):
        """Test loading an inference agent package and extracting its data."""
        # Create temporary agent data directory
        with tempfile.TemporaryDirectory() as data_dir:
            data_path = Path(data_dir)

            # Create test files in the data directory
            agent_file = data_path / "agent.py"
            agent_file.write_text(
                "# Inference agent implementation\nprint('Hello Agent')"
            )

            config_file = data_path / "config.json"
            config_file.write_text('{"param": "value"}')

            # Create metadata
            metadata = InferenceAgentMetadata(
                name="Test Inference Agent",
                version="1.0.0",
                authors=[
                    InferenceAgentAuthor(name="John Doe", email="john@example.com")
                ],
                maintainers=[
                    InferenceAgentMaintainer(
                        name="Jane Smith", email="jane@example.com"
                    )
                ],
                capabilities=frozenset(["text-generation", "summarization"]),
                compatible_targets=frozenset(["cpu", "gpu"]),
                compatible_tags=frozenset(["latest", "stable"]),
            )

            # Create package info
            package_info = InferenceAgentPackageInfo(
                metadata=metadata,
                agent_data_path=data_path,
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

                    # Load agent
                    loaded_metadata = agent_load(output_path, target_path)

                    # Verify metadata
                    assert isinstance(loaded_metadata, InferenceAgentMetadata)
                    assert loaded_metadata.name == "Test Inference Agent"

                    # Verify agent data was extracted
                    extracted_agent_file = target_path / "agent.py"
                    assert extracted_agent_file.exists()
                    assert (
                        extracted_agent_file.read_text()
                        == "# Inference agent implementation\nprint('Hello Agent')"
                    )

                    extracted_config_file = target_path / "config.json"
                    assert extracted_config_file.exists()
                    assert extracted_config_file.read_text() == '{"param": "value"}'
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
            with pytest.raises(MalformedAgentPackage):
                metadata_load(archive_path)
        finally:
            archive_path.unlink()

    def test_agent_load_malformed_package(self):
        """Test that loading agent fails for a malformed package."""
        # Create a temporary archive without metadata
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            archive_path = Path(f.name)

        # Create target directory
        with tempfile.TemporaryDirectory() as target_dir:
            target_path = Path(target_dir)

            try:
                # Create an empty archive
                with tarfile.open(archive_path, "w:gz") as archive:
                    pass

                # Try to load agent - should fail
                with pytest.raises(MalformedAgentPackage):
                    agent_load(archive_path, target_path)
            finally:
                archive_path.unlink()

    def test_agent_load_missing_data_folder(self):
        """Test that loading agent fails when data folder is missing from archive."""
        # Create temporary agent data directory
        with tempfile.TemporaryDirectory() as data_dir:
            data_path = Path(data_dir)

            # Create test file in the data directory
            agent_file = data_path / "agent.py"
            agent_file.write_text("# Inference agent implementation")

            # Create metadata
            metadata = InferenceAgentMetadata(
                name="Test Inference Agent",
                version="1.0.0",
                authors=[
                    InferenceAgentAuthor(name="John Doe", email="john@example.com")
                ],
                maintainers=[
                    InferenceAgentMaintainer(
                        name="Jane Smith", email="jane@example.com"
                    )
                ],
                capabilities=frozenset(["text-generation"]),
                compatible_targets=frozenset(["cpu"]),
                compatible_tags=frozenset(["latest"]),
            )

            # Create temporary metadata file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp_meta:
                meta_path = Path(tmp_meta.name)

            # Write metadata to temporary file
            from overity.exchange.inference_agent_package.metadata import to_file

            to_file(metadata, meta_path)

            # Create archive file
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                archive_path = Path(f.name)

            # Create target directory
            with tempfile.TemporaryDirectory() as target_dir:
                target_path = Path(target_dir)

                try:
                    # Create archive with only metadata, no data folder
                    with tarfile.open(archive_path, "w:gz") as archive:
                        archive.add(meta_path, arcname="agent-metadata.json")

                    # Try to load agent - should fail because data folder is missing
                    with pytest.raises(MalformedAgentPackage):
                        agent_load(archive_path, target_path)
                finally:
                    archive_path.unlink()
                    meta_path.unlink()
