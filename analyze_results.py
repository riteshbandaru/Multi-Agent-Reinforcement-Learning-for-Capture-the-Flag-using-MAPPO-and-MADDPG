import argparse
import os
from visualization import plot_training_curves

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory where the training log file is stored.")
    args = parser.parse_args()
    log_file = os.path.join(args.log_dir, "training_log.csv")
    plot_training_curves(log_file)

if __name__ == "__main__":
    main()
