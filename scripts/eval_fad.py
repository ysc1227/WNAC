from scripts.evaluation.fad import *
from scripts.evaluation.fad import evaluate_fad

if __name__ == "__main__":
    import argbind

    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate_fad()
