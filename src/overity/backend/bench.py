"""
Overity.ai bench management backend features
============================================

**September 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging

from pathlib import Path

from overity.storage.base import StorageBackend
from overity.storage.local import LocalStorage
from overity.errors import InvalidBenchSettingsError, BenchInstanciationError

log = logging.getLogger("backend.bench")


def list_benches(program_path: Path):
    """List available bench instanciations in program"""

    program_path = Path(program_path)

    log.info(f"List bench instanciations from program {program_path}")
    st = LocalStorage(program_path)

    benches, errors = st.benches()

    return benches, errors


def list_bench_abstractions(program_path: Path):
    """List available bench abstractions in program"""

    program_path = Path(program_path)

    log.info(f"List bench abstractions from program {program_path}")
    st = LocalStorage(program_path)

    abstractions, errors = st.bench_abstractions()

    return abstractions, errors


def load_bench_infos(program_path: Path, bench_slug: str):
    # Load program information
    program_path = Path(program_path)
    st = LocalStorage(program_path)

    bench = st.bench_load_infos(bench_slug)

    return bench


def load_bench_abstraction_infos(program_path: Path, slug: str):
    """Load bench abstraction information given its slug in specified programme"""
    # Load program information
    program_path = Path(program_path)
    st = LocalStorage(program_path)

    bench_abstraction = st.bench_abstraction_import_infos(slug)

    return bench_abstraction


def instanciate(program_path: Path, bench_slug: str, storage: StorageBackend):
    # Load program information
    program_path = Path(program_path)
    st = LocalStorage(program_path)

    # Import bench metadata
    bench_metadata = st.bench_load_infos(bench_slug)
    log.info(
        f"Instanciate bench '{bench_metadata.display_name}' ({bench_metadata.slug}) from {bench_metadata.abstraction_slug}"
    )

    # Import bench abstraction metadata
    bench_abstraction_metadata = st.bench_abstraction_import_infos(
        bench_metadata.abstraction_slug
    )

    # Import bench definitions
    BenchSettings, BenchDefinition = st.bench_abstraction_import_definitions(
        bench_metadata.abstraction_slug
    )

    # Parse bench settings
    try:
        bench_settings = BenchSettings(**bench_metadata.settings)
    except Exception as exc:
        raise InvalidBenchSettingsError(bench_slug, bench_metadata.settings, exc)

    # Instanciate bench
    try:
        bench_instance = BenchDefinition(
            settings=bench_settings,
            storage_backend=storage,
            abstraction_infos=bench_abstraction_metadata,
            instance_infos=bench_metadata,
        )
    except Exception as exc:
        raise BenchInstanciationError(bench_slug, bench_metadata.settings, exc)

    return bench_instance
