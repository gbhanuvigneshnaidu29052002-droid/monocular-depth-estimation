"""
Monocular Depth Estimation & Quadrant Risk Analysis Entrypoint
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🚀 Starting Monocular Depth Estimation & Quadrant Risk Analysis Pipeline...")
    import run_complete_pipeline

if __name__ == "__main__":
    main()
