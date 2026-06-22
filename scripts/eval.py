from scripts.evaluation.codec import *
from scripts.evaluation.codec import evaluate

if __name__ == "__main__":
    import argbind
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate()
