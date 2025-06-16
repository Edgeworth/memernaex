# Copyright 2025 Eliot Courtney.
from pathlib import Path

import cloup
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def read_free_energies_from_file(filepath: Path) -> np.ndarray | None:
    """Reads free energy values from the first column of a file."""
    energies = []
    print(f"--> Reading data from: {filepath}")
    try:
        with filepath.open() as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    parts = line.split()
                    energies.append(float(parts[0]))
                except (ValueError, IndexError):
                    print(f"    Warning: Could not parse line {line_num}. Skipping.")
                    continue
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None

    if not energies:
        print("Error: No valid energy values were found in the file.")
        return None

    print(f"--> Successfully read {len(energies)} energy values.")
    return np.array(energies)


@cloup.command(
    "generate-plots", help="Generates distribution plots from a file of free energy values."
)
@cloup.argument(
    "input_file",
    type=cloup.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Path to the input file. Each line should contain a free energy value in the first column",
)
@cloup.option(
    "--output-dir",
    default=Path("./energy_plots"),
    type=cloup.Path(
        dir_okay=True, file_okay=False, writable=True, resolve_path=True, path_type=Path
    ),
    help="Directory to save the output plots.",
)
@cloup.option(
    "--temperature",
    "-T",
    default=310.15,
    type=cloup.FloatRange(min=0),
    help="Temperature in Kelvin for Boltzmann calculations. Default: 310.15 K.",
)
@cloup.option(
    "--k-cal",
    default=1.987204259e-3,
    type=float,
    help="Boltzmann constant in kcal/(mol*K). Default value is standard.",
)
def plot_ensemble(input_file: Path, output_dir: Path, temperature: float, k_cal: float) -> None:
    """
    This script takes a file with free energy values, computes their statistical
    and Boltzmann distributions, and saves corresponding plots.
    """
    # --- 1. Setup and Data Loading ---
    output_dir.mkdir(parents=True, exist_ok=True)
    free_energies = read_free_energies_from_file(input_file)

    if free_energies is None:
        return  # Exit if file reading failed

    # --- 2. Define Constants and Bins ---
    beta = 1 / (k_cal * temperature)
    min_energy = np.floor(free_energies.min() * 10) / 10
    max_energy = np.ceil(free_energies.max() * 10) / 10
    bins = np.arange(min_energy, max_energy + 0.1, 0.1)

    # --- 3. Set Seaborn Style ---
    sns.set_theme(style="whitegrid")

    # --- 4. Plot Free Energy Distribution ---
    print("--> Generating free energy distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(x=free_energies, bins=bins, kde=True)
    plt.title(f"Free Energy Distribution (N={len(free_energies)})", fontsize=16)
    plt.xlabel("Free Energy (kcal/mol)", fontsize=12)
    plt.ylabel("Count", fontsize=12)

    dist_plot_path = output_dir / "free_energy_distribution.png"
    plt.savefig(dist_plot_path, bbox_inches="tight")
    plt.clf()
    print(f"    Saved plot to: {dist_plot_path}")

    # --- 5. Plot Boltzmann Distribution ---
    print("--> Generating Boltzmann distribution plot...")
    partition_function = np.sum(np.exp(-beta * free_energies))
    probabilities = np.exp(-beta * free_energies) / partition_function

    boltzmann_probabilities, _ = np.histogram(free_energies, bins=bins, weights=probabilities)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(10, 6))
    sns.barplot(x=np.round(bin_centers, 2), y=boltzmann_probabilities, color="skyblue")
    plt.title(f"Boltzmann Distribution (T={temperature}K)", fontsize=16)
    plt.xlabel("Free Energy (kcal/mol)", fontsize=12)
    plt.ylabel("Sum of Probabilities", fontsize=12)

    if len(bin_centers) > 20:
        plt.xticks(rotation=45, ha="right")
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=20))

    boltzmann_plot_path = output_dir / "boltzmann_distribution.png"
    plt.tight_layout()
    plt.savefig(boltzmann_plot_path)
    plt.close()
    print(f"    Saved plot to: {boltzmann_plot_path}")

    # --- 6. Final Output ---
    print("\n--- Summary ---")
    print(f"Partition Function (Q): {partition_function:.4g}")
    print(f"Sum of Probabilities: {np.sum(boltzmann_probabilities):.6f}")
    print("---------------")
