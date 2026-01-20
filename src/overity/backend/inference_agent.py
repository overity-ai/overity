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
from overity.exchange.inference_agent_package import package as agent_package

log = logging.getLogger("backend.inference_agents")


def list_agents(program_path: Path):
    """List the current available inference agents"""

    program_path = Path(program_path)

    log.info(f"List inference agents from program {program_path}")
    st = LocalStorage(program_path)

    agents, errors = st.inference_agents()

    return agents, errors


def list_agents_with_checksums(program_path: Path):
    """List the current available inference agents with their SHA256 checksums"""

    program_path = Path(program_path)

    log.info(f"List inference agents with checksums from program {program_path}")
    st = LocalStorage(program_path)

    agents, errors = st.inference_agents()

    # Add SHA256 checksums to each agent
    agents_with_checksums = []
    for ag_slug, ag_info in agents:
        try:
            agent_path = st._agent_path(ag_slug)
            if agent_path.is_file():
                sha256 = agent_package.package_sha256(agent_path)
                # Get hex digest for display
                checksum = sha256.hexdigest()
                agents_with_checksums.append((ag_slug, ag_info, checksum))
            else:
                agents_with_checksums.append((ag_slug, ag_info, "N/A"))
        except Exception as e:
            log.warning(f"Failed to compute checksum for agent {ag_slug}: {e}")
            agents_with_checksums.append((ag_slug, ag_info, "ERROR"))

    return agents_with_checksums, errors
