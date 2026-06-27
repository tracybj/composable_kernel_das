# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import subprocess


def __version__():
    dcu_baseline = "0.0.0+dcu"
    hash_width = 6
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )[:hash_width]
    except Exception:
        git_hash = "0" * hash_width
    return f"{dcu_baseline}.g{git_hash}"


_package_version = __version__()