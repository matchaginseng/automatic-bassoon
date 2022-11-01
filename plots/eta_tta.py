"""Generates ETA vs. TTA plot"""

import argparse
import matplotlib.pyplot as plt
from pathlib import Path

import zeus.analyze

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--power",
        nargs='*',
        type=Path,
        help="Paths to power limit files." # TODO: make this description better
    )
    parser.add_argument(
        "--time", 
        nargs='*', 
        type=Path,
        help="Paths to time files."
    )

    return parser.parse_args()

    

def main(args: argparse.Namespace) -> None:
    """Calculate average power and generate ETA vs. TTA plot"""
    # TODO: figure out how to get the batch size and power limit, lol
    power_files = args.power
    time_files = args.time

    # TODO: need to split()?
    energy = []
    avg_power = []
    tta = []
    eta = []

    for i in range(len(power_files)):
        power_path = power_files[i]
        time_path = time_files[i]
        print(f'power path: {power_path}, time path: {time_path}')
        energy.append(zeus.analyze.energy(power_path))
        avg_power.append(zeus.analyze.avg_power(power_path))
        tta.append(zeus.analyze.get_train_time(time_path))
        eta.append(avg_power[-1] * tta[-1])

    plt.plot(tta, eta, 'ro')
    plt.show()

if __name__ == "__main__":
    main(parse_args())
