#!/usr/bin/env python3

import pathlib
from typing import Dict, Any, List
import sys

sys.path.insert(0, "/home/muut/Documents/github/bachelorProefCode")
from benchkit.benchmark import Benchmark
from benchkit.platforms import Platform, get_current_platform
from benchkit.campaign import CampaignCartesianProduct, CampaignSuite
from benchkit.utils.dir import gitmainrootdir
from gpus import (
    get_gpu_runner,
    get_gpu_docker_platform_from,
    get_gpu_builder,
)
from smctrl import install_libsmctrl_from_src
from pathlib import Path

DOCKER = True
GUEST_SRC_DIR = "/home/user/src"


class SchedulerBenchmark(Benchmark):
    """Benchmark object for CUDA Scheduler comparisons."""

    def __init__(
        self,
        src_dir: Path,
        libsmctrl_dir: str,
        benchmark_file: str,
        scheduler_type: str,
        platform: Platform = None,
    ) -> None:
        super().__init__(
            command_wrappers=(),
            command_attachments=(),
            shared_libs=(),
            pre_run_hooks=(),
            post_run_hooks=(),
        )

        if platform is not None:
            self.platform = platform

        self._bench_src_path = pathlib.Path(src_dir)
        self._build_dir = self._bench_src_path / "build-benchkit"
        self._libsmctrl_dir = libsmctrl_dir
        self._benchmark_file = benchmark_file
        self._scheduler_type = scheduler_type
        self._results_path = self._bench_src_path / "results/scheduler_comparison"

    @property
    def bench_src_path(self) -> pathlib.Path:
        return self._bench_src_path

    @staticmethod
    def get_build_var_names() -> List[str]:
        return ["threads_per_block", "block_count", "tpc_denom"]

    @staticmethod
    def get_run_var_names() -> List[str]:
        return []

    @staticmethod
    def get_tilt_var_names() -> List[str]:
        return []

    def dependencies(self) -> List:
        return super().dependencies() + []

    def build_bench(
        self,
        threads_per_block: int,
        block_count: int,
        tpc_denom: int,
        **kwargs,
    ) -> None:
        build_dir = self._build_dir
        self.platform.comm.makedirs(path=build_dir, exist_ok=True)

        build_command = [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DSCHEDULER_TYPE={self._scheduler_type}",
            f"-DTHREADS_PER_BLOCK={threads_per_block}",
            f"-DTPC_SPLIT_DENOM={tpc_denom}",
            f"-DLIBSMCTRL_DIR={self._libsmctrl_dir}",
            f"{self.bench_src_path}",
        ]

        # Compile the benchmark with the specific scheduler
        self.platform.comm.shell(
            command=build_command,
            current_dir=build_dir,
            output_is_log=True,
        )

        self.platform.comm.shell(
            command=["make", "-j", f"{self.platform.nb_active_cpus()}"],
            current_dir=build_dir,
            output_is_log=True,
        )

    def clean_bench(self) -> None:
        pass

    def single_run(self, **kwargs) -> str:
        environment = self._preload_env(**kwargs)

        run_command = [f"./benchmark_{self._scheduler_type}"]

        wrapped_run_command, wrapped_environment = self._wrap_command(
            run_command=run_command,
            environment=environment,
            **kwargs,
        )

        output = self.run_bench_command(
            run_command=run_command,
            wrapped_run_command=wrapped_run_command,
            current_dir=self._build_dir,
            environment=environment,
            wrapped_environment=wrapped_environment,
            print_output=True,
        )

        return output

    def parse_output_to_results(self, command_output: str, **kwargs) -> Dict[str, Any]:
        """Parse the output from the benchmark run into structured results."""
        results = {
            "scheduler": self._scheduler_type,
            # "raw_output": command_output,
        }

        # Extract metrics from the output
        lines = command_output.strip().split("\n")
        for line in lines:
            if "Execution time:" in line:
                time_str = line.split(":")[-1].strip()
                results["execution_time"] = float(time_str.split()[0])
            if "Average latency:" in line:
                latency_str = line.split(":")[-1].strip()
                results["average_latency"] = float(latency_str.split()[0])
            if "Throughput:" in line:
                throughput_str = line.split(":")[-1].strip()
                results["throughput"] = float(throughput_str.split()[0])
            if "Jobs missed deadline:" in line:
                deadline_miss_str = line.split(":")[-1].strip()
                results["deadline_misses"] = int(deadline_miss_str.split()[0])
            if "Jobs completed:" in line:
                jobs_completed_str = line.split(":")[-1].strip()
                results["jobs_completed"] = int(jobs_completed_str.split()[0])

        return results


def get_docker_platform(
    host_src_dir: str,
) -> Platform:
    builder = get_gpu_builder()
    install_libsmctrl_from_src(
        builder=builder,
        workdir="/home/${USER_NAME}/workspace/libraries",
    )
    builder.build()
    runner = get_gpu_runner()

    docker_platform = get_gpu_docker_platform_from(
        runner=runner,
        host_src_dir=host_src_dir,
        guest_src_dir=GUEST_SRC_DIR,
    )

    return docker_platform


def scheduler_campaign(
    name: str = "scheduler_comparison",
    benchmark_file: str = "../../schedulerTestBenchmark.cu",
):
    """Create a campaign to compare different schedulers."""
    gmrd = gitmainrootdir()
    if DOCKER:
        platform = get_docker_platform(host_src_dir=str(gmrd))
        libsmctrl_dir = "/home/user/workspace/libraries/libsmctrl"
        src_dir = Path(GUEST_SRC_DIR)
    else:
        platform = get_current_platform()
        libsmctrl_dir = (
            "/home/muut/Documents/github/bachelorProefCode/commonLib/libsmctrl"
        )
        src_dir = gmrd

    # Define the schedulers to test
    schedulers = ["JLFP", "FCFS", "dumbScheduler"]

    # Create campaigns for each scheduler
    campaigns = []

    for scheduler in schedulers:
        bench = SchedulerBenchmark(
            src_dir=src_dir / "scheduling",
            libsmctrl_dir=libsmctrl_dir,
            benchmark_file=benchmark_file,
            scheduler_type=scheduler,
            platform=platform,
        )

        # Create a campaign for this scheduler
        campaign = CampaignCartesianProduct(
            name=f"{name}_{scheduler}",
            benchmark=bench,
            nb_runs=3,
            variables={
                "threads_per_block": [32],
                "block_count": [8],
                "tpc_denom": [2, 4],
            },
            constants={"scheduler": scheduler},
            debug=False,
            gdb=False,
            enable_data_dir=True,
            continuing=False,
            benchmark_duration_seconds=30,
        )

        campaigns.append(campaign)

    return campaigns


def main() -> None:
    campaigns = scheduler_campaign()

    # Create a campaign suite to run all campaigns
    suite = CampaignSuite(campaigns=campaigns)
    suite.print_durations()
    suite.run_suite()


if __name__ == "__main__":
    main()
