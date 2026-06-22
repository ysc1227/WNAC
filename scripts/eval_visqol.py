from scripts.evaluation.visqol import *
from scripts.evaluation.visqol import evaluate_visqol

if __name__ == "__main__":
    import argbind
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate_visqol()
