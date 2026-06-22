from scripts.training.aar import *
from scripts.training.aar import train

if __name__ == "__main__":
    import os
    import sys
    import argbind
    from scripts.training.aar import Accelerator
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
