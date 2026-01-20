"""
Overity.ai model backend features
=================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging

from pathlib import Path

from overity.storage.local import LocalStorage
from overity.exchange.model_package_v1 import package as ml_package

log = logging.getLogger("backend.model")


def list_models(program_path: Path | str):
    """List the current available models"""

    program_path = Path(program_path).resolve()

    log.info(f"List models from program {program_path}")
    st = LocalStorage(program_path)

    models, errors = st.models()

    return models, errors


def list_models_with_checksums(program_path: Path | str):
    """List the current available models with their SHA256 checksums"""

    program_path = Path(program_path).resolve()

    log.info(f"List models with checksums from program {program_path}")
    st = LocalStorage(program_path)

    models, errors = st.models()

    # Add SHA256 checksums to each model
    models_with_checksums = []
    for mod_slug, mod_info in models:
        try:
            model_path = st._model_path(mod_slug)
            if model_path.is_file():
                sha256 = ml_package.package_sha256(model_path)
                # Get hex digest for display
                checksum = sha256.hexdigest()
                models_with_checksums.append((mod_slug, mod_info, checksum))
            else:
                models_with_checksums.append((mod_slug, mod_info, "N/A"))
        except Exception as e:
            log.warning(f"Failed to compute checksum for model {mod_slug}: {e}")
            models_with_checksums.append((mod_slug, mod_info, "ERROR"))

    return models_with_checksums, errors
