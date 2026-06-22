from scripts.training.emac import *
from scripts.training.emac import train

if __name__ == "__main__":
    import argbind
    from scripts.training.emac import Accelerator
    args = argbind.parse_args()
    args["args.debug"] = True
    with argbind.scope(args):
        with Accelerator() as accel:
            train(args, accel)
