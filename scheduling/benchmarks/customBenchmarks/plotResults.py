import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob

results_directory = "/home/muut/Documents/github/bachelorProefCode/scheduling/benchmarks/customBenchmarks/results/temp/"


def load_and_aggregate_results(dir_path):
    """
    Loads benchmark results from all JSON files in a directory,
    aggregates them by scheduler, and calculates mean values.

    Args:
        dir_path (str): The path to the directory containing JSON files.

    Returns:
        pandas.DataFrame or None: An aggregated DataFrame with mean values
                                   per scheduler, or None if an error occurs.
    """
    all_data = []
    json_files = glob.glob(os.path.join(dir_path, "*.json"))  # Find all .json files

    if not json_files:
        print(f"Error: No JSON files found in directory: {dir_path}")
        return None

    print(f"Found {len(json_files)} JSON files. Processing...")

    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                # Handle cases where a file might contain a single result dict
                # or a list of results
                if isinstance(data, dict):
                    all_data.append(data)
                elif isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(
                        f"Warning: Skipping file {file_path} - unexpected data format."
                    )
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            continue  # Skip to the next file
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
            continue  # Skip to the next file
        except Exception as e:
            print(f"An unexpected error occurred processing {file_path}: {e}")
            continue  # Skip to the next file

    if not all_data:
        print("Error: No valid data could be loaded from the JSON files.")
        return None

    # Convert data to pandas DataFrame
    df = pd.DataFrame(all_data)

    # Check if required columns exist
    required_columns = ["scheduler", "throughput", "jobs_completed", "deadline_misses"]
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        print(
            f"Error: Missing required columns: {', '.join(missing)}. Ensure all JSON files contain these fields."
        )
        return None

    # Convert numeric columns to numeric type, coercing errors
    numeric_cols = ["throughput", "jobs_completed", "deadline_misses"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where conversion failed for any numeric column
    df.dropna(subset=numeric_cols, inplace=True)

    if df.empty:
        print("Error: No valid numeric data found for plotting after handling errors.")
        return None

    # Group by scheduler and calculate mean values
    # Use numpy's nanmean just in case, though dropna should handle NaNs
    results = (
        df.groupby("scheduler")
        .agg(
            avg_throughput=("throughput", np.mean),
            avg_jobs_completed=("jobs_completed", np.mean),
            avg_deadline_misses=("deadline_misses", np.mean),
        )
        .reset_index()
    )

    print("Aggregation complete.")
    print(results)  # Print the aggregated data for verification
    return results


def plot_aggregated_results(aggregated_df):
    """
    Plots the aggregated benchmark results (throughput, jobs completed,
    deadline misses) per scheduler.

    Args:
        aggregated_df (pandas.DataFrame): DataFrame with aggregated results.
    """
    if aggregated_df is None or aggregated_df.empty:
        print("No data to plot.")
        return

    schedulers = aggregated_df["scheduler"]
    metrics_to_plot = {
        "Average Throughput": aggregated_df["avg_throughput"],
        "Average Jobs Completed": aggregated_df["avg_jobs_completed"],
        "Average Deadline Misses": aggregated_df["avg_deadline_misses"],
    }

    num_plots = len(metrics_to_plot)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)

    # Ensure axes is always iterable, even if num_plots is 1
    if num_plots == 1:
        axes = [axes]

    colors = ["skyblue", "lightcoral", "lightgreen"]  # Colors for each plot

    for i, (title, data) in enumerate(metrics_to_plot.items()):
        ax = axes[i]
        ax.bar(schedulers, data, color=colors[i % len(colors)])
        ax.set_ylabel(title)
        ax.set_title(f"{title} by Scheduler")
        ax.grid(axis="y", linestyle="--")

    # Set common x-label and rotate ticks on the last plot
    axes[-1].set_xlabel("Scheduler")
    plt.xticks(rotation=45, ha="right")

    plt.suptitle(
        "Benchmark Comparison Across Schedulers", fontsize=16, y=1.02
    )  # Overall title
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout to prevent title overlap
    plt.savefig("aggregrate.png")


# --- Main execution ---
if __name__ == "__main__":
    aggregated_results = load_and_aggregate_results(results_directory)
    if aggregated_results is not None:
        plot_aggregated_results(aggregated_results)
