"""
Backend operations on reports
=============================

**May 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging

from pathlib import Path

from overity.model.report import MethodReportKind
from overity.storage.local import LocalStorage

log = logging.getLogger("backend.report")


def load(pdir: Path, kind: MethodReportKind, identifier: str):
    log.info(
        f"Load report '{identifier}' of kind '{kind.value}' from program stored at {pdir}"
    )

    st = LocalStorage(pdir)
    return st.report_load(kind, identifier)


def load_info(pdir: Path, kind: MethodReportKind, identifier: str):
    """This is a variant of the load function that should be used when requesting info for a report"""

    st = LocalStorage(pdir)
    return st.report_load(kind, identifier)


def list(pdir: Path, kind: MethodReportKind):
    log.info(f"Get list of '{kind.value}' reports for program in '{pdir}'")

    st = LocalStorage(pdir)
    return st.reports_list(kind)


def remove(pdir: Path, kind: MethodReportKind, identifier: str):
    st = LocalStorage(pdir)
    st.report_remove(kind, identifier)
