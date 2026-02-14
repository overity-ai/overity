"""
Overity.ai inference agents backend features
============================================

**August 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging
from pathlib import Path

from overity.storage.local import LocalStorage

log = logging.getLogger("backend.inference_agents")


def list_agents(program_path: Path):
    """List the current available inference agents"""

    program_path = Path(program_path)

    log.info(f"List inference agents from program {program_path}")
    st = LocalStorage(program_path)

    agents, errors = st.inference_agents()

    return agents, errors
