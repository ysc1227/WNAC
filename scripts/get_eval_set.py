from scripts.data.export_eval_set import *
from scripts.data.export_eval_set import save_test_set

if __name__ == "__main__":
    import argbind
    from scripts.training.emac import Accelerator
    args = argbind.parse_args()
    with argbind.scope(args):
        with Accelerator() as accel:
            save_test_set(args, accel)
