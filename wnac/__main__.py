import sys

import argbind

from wnac.utils import download
from wnac.utils.decode import decode
from wnac.utils.encode import encode
from wnac.utils.recon import recon

STAGES = ["encode", "decode", "download", "recon"]


def run(stage: str):
    """Run stages.

    Parameters
    ----------
    stage : str
        Stage to run
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown command: {stage}. Allowed commands are {STAGES}")
    stage_fn = globals()[stage]

    if stage == "download":
        stage_fn()
        return

    stage_fn()


if __name__ == "__main__":
    group = sys.argv.pop(1)
    args = argbind.parse_args(group=group)

    with argbind.scope(args):
        run(group)