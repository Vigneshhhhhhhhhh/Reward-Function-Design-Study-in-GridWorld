import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization.generate_figs import run_visualization_suite

if __name__ == "__main__":
    print("======================================================")
    print(" Generating Visualizations for Result JSONs")
    print("======================================================")
    run_visualization_suite()
