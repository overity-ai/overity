"""
Overity.ai dataset backend features
===================================

**August 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging
from pathlib import Path

from overity.storage.local import LocalStorage
from overity.exchange.dataset_package import package as dataset_package

log = logging.getLogger("backend.dataset")


def list_datasets(program_path: Path):
    """List the current available datasets"""

    program_path = Path(program_path)

    log.info(f"List avialalbe datasets from program {program_path}")
    st = LocalStorage(program_path)

    datasets, errors = st.datasets()

    return datasets, errors


def list_datasets_with_checksums(program_path: Path):
    """List the current available datasets with their SHA256 checksums"""

    program_path = Path(program_path)

    log.info(f"List datasets with checksums from program {program_path}")
    st = LocalStorage(program_path)

    datasets, errors = st.datasets()

    # Add SHA256 checksums to each dataset
    datasets_with_checksums = []
    for ds_slug, ds_info in datasets:
        try:
            dataset_path = st._dataset_path(ds_slug)
            if dataset_path.is_file():
                sha256 = dataset_package.package_sha256(dataset_path)
                # Get hex digest for display
                checksum = sha256.hexdigest()
                datasets_with_checksums.append((ds_slug, ds_info, checksum))
            else:
                datasets_with_checksums.append((ds_slug, ds_info, "N/A"))
        except Exception as e:
            log.warning(f"Failed to compute checksum for dataset {ds_slug}: {e}")
            datasets_with_checksums.append((ds_slug, ds_info, "ERROR"))

    return datasets_with_checksums, errors
