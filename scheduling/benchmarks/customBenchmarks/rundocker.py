#!/usr/bin/env python3
# Copyright (C) 2025 Vrije Universiteit Brussel. All rights reserved.
# SPDX-License-Identifier: MIT

from gpus import get_gpu_builder, install_libsmctrl_from_src, get_gpu_runner, DockerRunner
from benchkit.utils.dir import gitmainrootdir


if __name__ == "__main__":
    builder = get_gpu_builder()
    install_libsmctrl_from_src(
        builder=builder,
        workdir="/home/${USER_NAME}/workspace/libraries",
    )
    builder.build()

    runner = get_gpu_runner(workdir="/home/user/src/scheduling")
    runner |= DockerRunner(volumes={f"{gitmainrootdir()}": "/home/user/src"})

    runner.run()
