#!/usr/bin/env python3

import pathlib
from typing import Dict, Any, List
import sys

sys.path.insert(0, "/home/muut/Documents/github/bachelorProefCode")
from benchkit.benchmark import Benchmark
from benchkit.campaign import Campaign
from benchkit.platforms import Platform, get_current_platform
from benchkit.campaign import CampaignCartesianProduct, CampaignSuite
from benchkit.utils.dir import gitmainrootdir
from pathlib import Path
from gpugangbenchutil.docker import get_docker_platform, GUEST_SRC_DIR
from gpugangbenchutil.benchmark import SchedulerBenchmark


DOCKER = True


def scheduler_campaign(
    name: str = "scheduler_comparison",
    benchmark_file: str = "../../schedulerTestBenchmark.cu",
) -> List[CampaignCartesianProduct]:
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
