"""
Utilities to dump environment information
=========================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging

from overity.errors import NoBenchDefinedError

from overity.model.report import MethodExecutionStage

log = logging.getLogger("backend.environment")


def platform_info():
    import platform

    uname_info = platform.uname()

    return {
        "hostname": uname_info.node,
        "machine": uname_info.machine,
        "os_system": uname_info.system,
        "os_release": uname_info.release,
        "os_version": uname_info.version,
    }


def installed_packages():
    """List installed packages in current environment using pip freeze"""
    import pkg_resources

    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(
        ["%s==%s" % (i.key, i.version) for i in installed_packages]
    )

    return installed_packages_list


def bench() -> str:
    """Get used bench name through environment variable"""

    import os

    value = os.getenv("OVERITY_BENCH")
    if value is None:
        raise NoBenchDefinedError()

    return value


def execution_stage():
    """Detect the execution stage

    This is priparmy done through the OVERITY_STAGE environment variable. This variable
    may be set by the end-user or through the overity run CLI.
    """

    import os

    stage_str = os.getenv("OVERITY_STAGE") or "preview"

    try:
        stage = MethodExecutionStage(stage_str)
    except ValueError:  # Value is unkonwn
        log.warning(
            f"Invalid OVERITY_STAGE environment variable value: '{stage_str!r}'. Defaulting to preview mode"
        )
        stage = MethodExecutionStage.Preview

    return stage
