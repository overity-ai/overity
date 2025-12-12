"""
ML Model packaging tools
========================

**December 2024**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import json
import tarfile
import tempfile
import hashlib

from pathlib import Path

from overity.exchange.model_package_v1 import metadata as ml_metadata

from overity.model.ml_model.metadata import MLModelMetadata
from overity.model.ml_model.package import MLModelPackage

from overity.errors import MalformedModelPackage


# TODO # Merge with one used in inference agent package
def package_sha256(path: Path):
    path = Path(path)

    with open(path, "rb") as fhandle:
        digest = hashlib.file_digest(fhandle, "sha256")

    return digest


def package_archive_create(model_data: MLModelPackage, output_path: Path):
    output_path = Path(output_path)

    with tempfile.NamedTemporaryFile(delete_on_close=False) as fhandle:
        # Encode metadata to JSON temporary file
        fhandle.close()  # File will be reopened by exchange encoding
        ml_metadata.to_file(model_data.metadata, fhandle.name)

        # Create output archive
        with tarfile.open(output_path, "w:gz") as archive:
            archive.add(fhandle.name, arcname="model-metadata.json")
            archive.add(
                model_data.model_file_path, arcname=model_data.metadata.model_file
            )

            if model_data.example_implementation_path is not None:
                archive.add(model_data.example_implementation_path, "inference-example")

    # -> fhandle file is removed automatically when exiting the with... clause
    return package_sha256(output_path)


def _process_metadata(archive_path: Path, tf: tarfile.TarFile):
    """Utility function to load metadata from archive file"""

    try:
        info_json = tf.getmember("model-metadata.json")
    except KeyError:
        raise MalformedModelPackage(archive_path, "No model-metadata.json file")

    with tf.extractfile(info_json) as fhandle:
        data = json.load(fhandle)

    return ml_metadata.from_dict(data)


def _process_model_file(
    archive_path: Path,
    tf: tarfile.TarFile,
    metadata: MLModelMetadata,
    target_folder: Path,
):
    target_folder = Path(target_folder)

    try:
        model_file = tf.getmember(metadata.model_file)
    except KeyError:
        raise MalformedModelPackage(
            archive_path,
            f"Indicated model_file, '{metadata.model_file}', is not found in archive",
        )

    with (
        open(target_folder / metadata.model_file, "wb") as ftarget,
        tf.extractfile(model_file) as fmod,
    ):
        ftarget.write(fmod.read())
        ftarget.flush()


def metadata_load(archive_path: Path) -> MLModelMetadata:
    """Load only metadata from archive path"""

    archive_path = Path(archive_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        meta = _process_metadata(archive_path, archive)

    return meta


def model_load(archive_path: Path, target_folder: Path) -> MLModelMetadata:
    """Load model metadata from archive, and load model implementation in ftarget"""

    archive_path = Path(archive_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        meta = _process_metadata(archive_path, archive)
        _process_model_file(archive_path, archive, meta, target_folder)

    return meta
