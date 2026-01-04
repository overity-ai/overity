"""
Overity.ai methods backend features
===================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging
from pathlib import Path

from overity.model.general_info.method import MethodKind
from overity.storage.local import LocalStorage


log = logging.getLogger("backend.methods")


def list_topt_methods(program_path: Path | str):
    """List the current available training/optimization methods from the given program path"""

    program_path = Path(program_path).resolve()

    log.info(f"List training/optimization methods from program in {program_path}")
    st = LocalStorage(program_path)
    methods, errors = st.training_optimization_methods()

    return methods, errors


def list_measurement_qualification_methods(program_path: Path | str):
    """List the current available measurement/qualification methods from the given program path"""

    program_path = Path(program_path).resolve()

    log.info(f"List measurement/qualification methods from program in {program_path}")
    st = LocalStorage(program_path)
    methods, errors = st.measurement_qualification_methods()

    return methods, errors


def find_method_path(program_path: Path | str, kind: MethodKind, slug: str) -> Path:
    """Find the requested method script file"""

    program_path = Path(program_path)

    log.info(f"Looking for script file for method {slug!r} of kind {kind.value!r}")
    st = LocalStorage(program_path)
    path = st.get_method_path(kind, slug)

    return path
