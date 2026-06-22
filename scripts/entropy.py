from scripts.evaluation.codebook_entropy import *
from scripts.evaluation.codebook_entropy import main

if __name__ == "__main__":
    import argbind
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
