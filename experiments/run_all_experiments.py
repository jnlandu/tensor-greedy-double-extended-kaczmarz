#!/usr/bin/env python3
"""
run_all_experiments.py
======================
Thin orchestrator for all six TGDBEK numerical experiments.

Usage:
  python run_all_experiments.py              # run all examples
  python run_all_experiments.py --example 2  # run one example (1, 2, 3, 5, 6)
  python run_all_experiments.py --skip-images

Each example can also be run standalone:
  python ex1_dense.py
  python ex2_sparse.py
  python ex3_color_image.py
  python ex5_gray_image.py
  python ex6_eta.py

Figures  → experiments/figures/
Tables   → run generate_tables_pdf.py  →  experiments/tables/tables.pdf
"""
import argparse, sys
from utils import OUTDIR

import ex1_dense
import ex2_sparse
import ex3_color_image
import ex5_gray_image
import ex6_eta

DISPATCH = {
    1: ex1_dense.main,
    2: ex2_sparse.main,
    3: ex3_color_image.main,
    5: ex5_gray_image.main,
    6: ex6_eta.main,
}


def main():
    parser = argparse.ArgumentParser(description="Run TGDBEK experiments")
    parser.add_argument("--example", type=int, default=0,
                        help="Run only example N (1, 2, 3, 5, 6). 0 = run all.")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip image deblurring examples (4.3, 4.5)")
    args = parser.parse_args()

    if args.example:
        if args.example not in DISPATCH:
            print(f"Unknown example {args.example}. Choose one of {sorted(DISPATCH)}.")
            sys.exit(1)
        DISPATCH[args.example]()
    else:
        for num, fn in DISPATCH.items():
            if args.skip_images and num in (3, 5):
                print(f"  Skipping Example 4.{num} (--skip-images)")
                continue
            fn()

    print(f"\nAll figures saved to: {OUTDIR}")
    print("Run generate_tables_pdf.py to rebuild experiments/tables/tables.pdf\n")


if __name__ == "__main__":
    main()
