"""
Inference agent packaging tools
===============================

**July 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import tempfile
import tarfile
import json

from pathlib import Path

from overity.model.inference_agent.metadata import InferenceAgentMetadata
from overity.model.inference_agent.package import InferenceAgentPackageInfo

from overity.exchange.inference_agent_package import metadata as agent_metadata
from overity.exchange import integrity

from overity.errors import MalformedAgentPackage


####################################################
# Create package
####################################################


def package_archive_create(agent_data: InferenceAgentPackageInfo, output_path: Path):
    output_path = Path(output_path)

    with tempfile.NamedTemporaryFile(delete_on_close=False) as fhandle:
        # Encode metadata to JSON temporary file
        fhandle.close()  # File will be reopened by exchange encoding
        agent_metadata.to_file(agent_data.metadata, Path(fhandle.name))

        # Create output archive
        with tarfile.open(output_path, "w:gz") as archive:
            archive.add(fhandle.name, arcname="agent-metadata.json")
            archive.add(agent_data.agent_data_path, arcname="data")

    # -> fhandle file is removed automatically when exiting the with... clause
    return integrity.file_sha256(output_path)


####################################################
# Open package
####################################################


def _process_metadata(archive_path: Path, tf: tarfile.TarFile):
    """Utility function to load metadata from archive file"""

    try:
        info_json = tf.getmember("agent-metadata.json")

    except KeyError:
        raise MalformedAgentPackage(archive_path, "No agent-metadata.json file")

    fhandle = tf.extractfile(info_json)
    if fhandle is None:
        raise MalformedAgentPackage(
            archive_path, "Cannot extract agent-metadata.json file"
        )

    with fhandle as fh:
        data = json.load(fh)

    return agent_metadata.from_dict(data)


def _process_agent_data(archive_path: Path, tf: tarfile.TarFile, target_folder: Path):
    """Utility function to extract agent data from package archive"""

    agent_data_member = "data"
    agent_data_prefix = agent_data_member + "/"

    try:
        data = tf.getmember(agent_data_member)
        if not data.isdir():
            raise MalformedAgentPackage(archive_path, "data member is not a directory")

    except KeyError:
        raise MalformedAgentPackage(archive_path, "No data/ folder in package")

    # List of files in data folder
    data_files = filter(
        lambda x: x.name.startswith(agent_data_prefix) and x.isfile(), tf.getmembers()
    )

    # TODO # Maybe refactor elsewhere as multiple archive formats (datset for instance) may exist?
    def __process_member(target_folder, x: tarfile.TarInfo):
        # Remove data/ prefix from the name
        target_name = x.name[len(agent_data_prefix) :]

        if target_name:  # -> Skip if target_name is empty (the data directory itself)
            target_path = target_folder / target_name

            # Create parent directories
            # TODO # Ensure still in target folder to avoid relative path stuff?
            target_path.parent.mkdir(parents=True, exist_ok=True)

            source = tf.extractfile(x)
            if source is None:
                raise MalformedAgentPackage(
                    archive_path, f"Cannot extract file {member.name}"
                )

            with source as src, open(target_path, "wb") as target:
                # TODO # check if streaming or in-memory stuff to avoid bad RAM stuff if big file
                target.write(src.read())

    for member in data_files:
        __process_member(target_folder, member)


def metadata_load(archive_path: Path) -> InferenceAgentMetadata:
    """Load only agent metadata from archive path"""

    archive_path = Path(archive_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        meta = _process_metadata(archive_path, archive)

    return meta


def agent_load(archive_path: Path, target_folder: Path) -> InferenceAgentMetadata:
    """Load agent metadata from archive, and extract its data to target folder"""

    archive_path = Path(archive_path)
    target_folder = Path(target_folder)

    with tarfile.open(archive_path, "r:gz") as archive:
        meta = _process_metadata(archive_path, archive)
        _process_agent_data(archive_path, archive, target_folder)

    return meta
