"""
Dataset packaging tools
=======================

**August 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import tempfile
import tarfile
import json

from pathlib import Path

from overity.model.dataset.metadata import DatasetMetadata
from overity.model.dataset.package import DatasetPackageInfo

from overity.exchange.dataset_package import metadata as dataset_metadata
from overity.exchange import integrity

from overity.errors import MalformedDatasetPackage


####################################################
# Create package
####################################################


def package_archive_create(dataset_data: DatasetPackageInfo, output_path: Path):
    output_path = Path(output_path)

    with tempfile.NamedTemporaryFile(delete_on_close=False) as fhandle:
        # Encode metadata to JSON temporary file
        fhandle.close()  # File will be reopened by exchange encoding
        dataset_metadata.to_file(dataset_data.metadata, fhandle.name)

        # Create output archive
        with tarfile.open(output_path, "w:gz") as archive:
            archive.add(fhandle.name, arcname="dataset-metadata.json")
            archive.add(dataset_data.dataset_data_path, arcname="data")

    # -> fhandle file is removed automatically when exiting the with... clause
    return integrity.file_sha256(output_path)


####################################################
# Open package
####################################################


def _process_metadata(archive_path: Path, tf: tarfile.TarFile):
    """Utility function to load metadata from archive file"""

    try:
        info_json = tf.getmember("dataset-metadata.json")
    except KeyError:
        raise MalformedDatasetPackage(archive_path, "No dataset-metadata.json file")

    with tf.extractfile(info_json) as fhandle:
        data = json.load(fhandle)

    return dataset_metadata.from_dict(data)


def _process_dataset_data(archive_path: Path, tf: tarfile.TarFile, target_folder: Path):
    """Utility function to extract the dataset data from package archive"""

    try:
        _ = tf.getmember("data")
    except KeyError:
        raise MalformedDatasetPackage(archive_path, "No data/ folder in package")

    # List of files in data folder
    data_files = filter(lambda x: x.name.startswith("data/"), tf.getmembers())

    tf.extractall(
        target_folder,
        members=list(
            data_files,
        ),
        filter="data",
    )


def metadata_load(archive_path: Path) -> DatasetMetadata:
    """Load dataset metadata from archive path"""

    archive_path = Path(archive_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        meta = _process_metadata(archive_path, archive)

    return meta


def dataset_load(archive_path: Path, target_folder: Path) -> DatasetPackageInfo:
    """Load dataset metadata from archive, and extract its data to target folder"""

    archive_path = Path(archive_path)
    target_folder = Path(target_folder)

    with tarfile.open(archive_path, "r:gz") as archive:
        meta = _process_metadata(archive_path, archive)
        _process_dataset_data(archive_path, archive, target_folder)

    return meta
