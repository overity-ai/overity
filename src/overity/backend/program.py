"""
Overity.ai program backend features
===================================

**March 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com): Initial design

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

import logging
from pathlib import Path

from overity.errors import ProgramNotFound
from overity.storage.local import LocalStorage
from overity.exchange import program_toml

from overity.model.general_info.program import ProgramInitiator, ProgramInfo

from overity.utils.path import iter_path

from datetime import date

log = logging.getLogger("backend.program")


########################################
# Find program.toml file
########################################


def is_program(path: Path):
    """Indicates if the current folder is the root folder of a program"""
    return (path / "program.toml").is_file()


def find_current(start_path: Path):
    log.debug(f"Search program root folder starting from {start_path}")

    if is_program(start_path):
        return start_path

    else:
        for subpath in iter_path(start_path.parent):
            log.debug(f"Check parent path: {subpath}")

            if is_program(subpath):
                return subpath

        raise ProgramNotFound(start_path=start_path, recursive=True)


def infos(path: Path):
    """Load program information"""

    path = Path(path).resolve()

    log.info(f"Load program information from {path}")

    st = LocalStorage(path)

    return st.program_info()


########################################
# Initialize local program
########################################


def initialize(
    dest_path: Path,
    slug: str,
    display_name: str,
    initiator_name: str,
    initiator_email: str,
    initiator_role: str,
    date_created: date,
    description: str | None = None,
):
    log.info(f"Initialize program in {dest_path}")

    # Create program infos
    prginfo = ProgramInfo(
        slug=slug,
        display_name=display_name,
        date_created=date_created,
        initiator=ProgramInitiator(
            name=initiator_name,
            email=initiator_email,
            role=initiator_role,
        ),
        description=description,
        active=True,
    )

    # Initialize local storage
    log.debug("-> Initialize local storage folder structure")
    st = LocalStorage(dest_path)
    st.initialize()  # Create folder structure

    # Create program.toml file
    log.debug("-> Create program.toml file")
    program_toml.to_file(prginfo, toml_path=st.program_info_path)
