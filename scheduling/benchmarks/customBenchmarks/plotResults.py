import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re


def clean_csv(input_file, output_file):
    """Fix the CSV format by properly handling the raw_output field with newlines."""
    with open(input_file, "r") as f:
        content = f.read()

    # Extract header and data lines
    lines = content.split("\n")
    header = None
    cleaned_lines = []

    for i, line in enumerate(lines):
        if line.startswith("#"):
            continue  # Skip comment lines
        elif line.startswith("experiment_name;"):
            header = line
            cleaned_lines.append(line)
        elif ";" in line:
            # This is a data row start
            # Extract values from this line and subsequent lines until we find metrics
            current_line = line
            j = i + 1
            while j < len(lines) and not lines[j].startswith(";"):
                current_line += "\\n" + lines[j]
                j = j + 1

            # Check if we found the metrics at the end
            if j < len(lines) and lines[j].startswith(";"):
                metrics = lines[j].strip()
                # Combine everything properly
                cleaned_line = current_line.replace("\n", "\\n") + metrics
                cleaned_lines.append(cleaned_line)

    # Write cleaned CSV
    with open(output_file, "w") as f:
        f.write("\n".join(cleaned_lines))

    return output_file


def extract_metrics_directly(input_file):
    """Extract metrics directly from the raw file using regex."""
    with open(input_file, "r") as f:
        content = f.read()

    # First get the header to know column names
    match = re.search(r"^experiment_name.*?throughput", content, re.MULTILINE)
    if not match:
        print(f"Error: Could not find CSV header in {input_file}")
        return None

    header = match.group(0).split(";")

    # Define patterns to extract values
    patterns = {
        "scheduler": r"Scheduler: (\w+)",
        "threads_per_block": r"Threads Per Block: (\d+)",
        "block_count": r"Block Count: (\d+)",
        "data_size": r"Data Size: (\d+)",
        "execution_time": r"Execution time: ([\d\.]+)",
        "average_latency": r"Average latency: ([\d\.]+)",
        "throughput": r"Throughput: ([\d\.]+)",
    }

    data = []

    # Find each benchmark run
    run_pattern = r"Starting benchmark.*?jobs/second"
    for i, run_match in enumerate(re.finditer(run_pattern, content, re.DOTALL)):
        run_text = run_match.group(0)
        row = {}

        # Extract each metric
        for key, pattern in patterns.items():
            match = re.search(pattern, run_text)
            if match:
                try:
                    row[key] = (
                        float(match.group(1))
                        if key not in ["scheduler"]
                        else match.group(1)
                    )
                except ValueError:
                    row[key] = match.group(1)

        # Add rep number (run index)
        row["rep"] = i + 1
        data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df


def plot_benchmark_results(file_path=None):
    """Plot benchmark results from CSV files."""
    # Determine if we're processing a directory or specific file(s)
    if file_path is None:
        directory = os.getcwd()
        csv_files = glob.glob(f"{directory}/*.csv")
        if not csv_files:
            print(f"No CSV files found in {directory}")
            return
    elif os.path.isdir(file_path):
        directory = file_path
        csv_files = glob.glob(f"{file_path}/*.csv")
        if not csv_files:
            print(f"No CSV files found in {file_path}")
            return
    elif os.path.isfile(file_path) and file_path.endswith(".csv"):
        directory = os.path.dirname(file_path)
        csv_files = [file_path]
    else:
        print(f"Invalid path: {file_path}")
        return

    print(f"\nProcessing {len(csv_files)} CSV file(s)")

    # Read CSV files using direct extraction
    dfs = []
    for file in csv_files:
        try:
            print(f"Processing {os.path.basename(file)}")
            # Extract metrics directly from file
            df = extract_metrics_directly(file)

            if df is not None and not df.empty:
                dfs.append(df)
                print(f"Successfully read {len(df)} rows from {os.path.basename(file)}")
            else:
                print(f"No valid data extracted from {os.path.basename(file)}")
        except Exception as e:
            print(f"Error processing {os.path.basename(file)}: {e}")

    if not dfs:
        print("No valid data found in CSV files.")
        return

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined data: {len(combined_df)} rows")

    # Print available columns for plotting
    print("\nAvailable columns for plotting:")
    for col in combined_df.columns:
        print(f"- {col}")

    # Create plots directory
    plots_dir = os.path.join(directory, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Check which columns are available for plotting
    metric_columns = []
    for col in ["execution_time", "average_latency", "throughput"]:
        if col in combined_df.columns:
            metric_columns.append(col)

    x_columns = []
    for col in ["threads_per_block", "block_count", "data_size"]:
        if col in combined_df.columns:
            x_columns.append(col)

    # Plot bar charts for metrics by scheduler
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=combined_df, x="scheduler", y="execution_time", errorbar=None)
    plt.title("Execution Time by Scheduler")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "execution_time_by_scheduler.png"))
    plt.close()
    print(f"Created plot: execution_time_by_scheduler.png")

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=combined_df, x="scheduler", y="throughput", errorbar=None)
    plt.title("Throughput by Scheduler")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "throughput_by_scheduler.png"))
    plt.close()
    print(f"Created plot: throughput_by_scheduler.png")

    # Only plot line charts if we have multiple configurations
    if len(combined_df[x_columns[0]].unique()) > 1:
        for x_col in x_columns:
            for y_col in metric_columns:
                plt.figure(figsize=(12, 8))
                sns.lineplot(
                    data=combined_df, x=x_col, y=y_col, hue="scheduler", markers=True
                )
                plt.title(
                    f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}"
                )
                plt.grid(True, linestyle="--", alpha=0.7)
                plt.legend(
                    title="Scheduler", bbox_to_anchor=(1.05, 1), loc="upper left"
                )
                plt.xlabel(x_col.replace("_", " ").title())
                plt.ylabel(y_col.replace("_", " ").title())
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{y_col}_vs_{x_col}.png"))
                plt.close()
                print(f"Created plot: {y_col}_vs_{x_col}.png")
    else:
        print("\nNote: Only one configuration found. Not creating line plots.")

    # Inform user where plots are saved
    print(f"\nPlots saved to: {plots_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        plot_benchmark_results(path)
    else:
        plot_benchmark_results()
