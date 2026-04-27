import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Define file paths
train_mse_path = "/home/jhong90/github_proj/flare_surya/results/train/xrs/train_mse.csv"
validation_mse_path = (
    "/home/jhong90/github_proj/flare_surya/results/train/xrs/validation_mse.csv"
)
output_dir = (
    "/home/jhong90/github_proj/flare_surya/results/plots"  # Directory to save plots
)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load data
try:
    df_train = pd.read_csv(train_mse_path)
    df_val = pd.read_csv(validation_mse_path)
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    exit()

# --- Dynamically identify experiments and MSE columns ---


def extract_experiment_mse_data(df, file_path, mse_type):
    """
    Extracts MSE data for specified experiments from a DataFrame.
    mse_type should be 'train/mse' or 'val/mse'.
    Returns a dictionary mapping display_name to (x_axis_data, mse_series).
    """
    experiment_data = {}
    # Regex to find experiment identifiers like '_test12', '_test13', '_test14'
    exp_id_pattern = re.compile(r"(_test\d+)")

    # Pattern for the MSE column: something + exp_id + " - " + mse_type
    # Use re.escape for the mse_type to handle potential special characters if any
    mse_col_pattern = re.compile(rf".* - {re.escape(mse_type)}$")

    potential_mse_cols = [col for col in df.columns if mse_col_pattern.search(col)]

    # Process each potential MSE column
    for col in potential_mse_cols:
        match = exp_id_pattern.search(col)
        if match:
            exp_id = match.group(1)  # e.g., '_test12'
            # Use a simplified display name for plotting legend
            display_name = f"Exp {exp_id.replace('_', '')}"

            # Extract MSE values, dropping any NaNs
            mse_series = df[col].dropna()

            if not mse_series.empty:
                # Check for corresponding x-axis column
                # Try "epoch" first, fallback to "trainer/global_step"
                x_axis_col = "epoch" if "epoch" in df.columns else "trainer/global_step"
                if x_axis_col in df.columns:
                    x_axis_data = df[x_axis_col].dropna()
                    # Ensure x-axis data length matches MSE series length before storing
                    # We take the minimum length of the series and the corresponding x-axis slice
                    min_len = min(len(x_axis_data), len(mse_series))
                    if min_len > 0:
                        experiment_data[display_name] = (
                            x_axis_data.iloc[:min_len],
                            mse_series.iloc[:min_len],
                        )
                    else:
                        print(
                            f"Warning: No common data points found for {col} in {file_path} after matching lengths."
                        )
                else:
                    print(
                        f"Warning: X-axis column '{x_axis_col}' not found for {col} in {file_path}."
                    )
        else:
            print(
                f"Warning: Could not find experiment identifier in MSE column: {col} in {file_path}"
            )

    return experiment_data


# Get training data
train_results = extract_experiment_mse_data(df_train, train_mse_path, "train/mse")
# Get validation data
val_results = extract_experiment_mse_data(df_val, validation_mse_path, "val/mse")

# --- Determine minimum steps and truncate ---
min_train_len = float("inf")
if train_results:
    min_train_len = min(len(mse_series) for _, mse_series in train_results.values())

min_val_len = float("inf")
if val_results:
    min_val_len = min(len(mse_series) for _, mse_series in val_results.values())

# Truncate all data to the minimum length
truncated_train_results = {}
if train_results and min_train_len != float("inf"):
    for display_name, (x_axis, mse_series) in train_results.items():
        truncated_train_results[display_name] = (
            x_axis.iloc[:min_train_len],
            mse_series.iloc[:min_train_len],
        )

truncated_val_results = {}
if val_results and min_val_len != float("inf"):
    for display_name, (x_axis, mse_series) in val_results.items():
        truncated_val_results[display_name] = (
            x_axis.iloc[:min_val_len],
            mse_series.iloc[:min_val_len],
        )


# --- Helper Function ---
def plot_mse_on_axis(ax, results, ordered_exp_ids, test_condition, title, xlabel):
    """Plots MSE data on a given axis, limited to epoch 60."""
    # Define styles for experiments to be distinct in B&W and colorblind-friendly
    styles = {
        "test13": {"linestyle": "-", "marker": "o", "color": "#0072B2"}, # Blue
        "test14": {"linestyle": "--", "marker": "s", "color": "#D55E00"}, # Vermillion
        "test12": {"linestyle": ":", "marker": "^", "color": "#009E73"},  # Bluish Green
    }

    for exp_id in ordered_exp_ids:
        display_name = f"Exp {exp_id}"
        if display_name in results:
            x_axis, mse_series = results[display_name]
            
            # Filter data to include only epochs <= 60
            mask = x_axis <= 60
            x_axis_filtered = x_axis[mask]
            mse_series_filtered = mse_series[mask]
            
            label = test_condition.get(exp_id, display_name)
            
            # Apply style
            style = styles.get(exp_id, {})
            # Use markerevery to not overwhelm the plot if there are many points
            lines = ax.plot(
                x_axis_filtered, 
                mse_series_filtered, 
                label=label, 
                linewidth=2.5, 
                **style,
                markersize=6
            )
            lines[0].set_markevery(max(1, len(x_axis_filtered) // 10))
            
    ax.set_title(title, fontsize=24, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=20)
    ax.legend(fontsize=18, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=16)


# --- Create and Save Combined Plot ---
test_condition = {"test13": "Small", "test14": "Medium", "test12": "Large"}
ordered_exp_ids = ["test13", "test14", "test12"]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
ax_train, ax_val = axes

# Plot Training MSE
if truncated_train_results:
    plot_mse_on_axis(
        ax_train,
        truncated_train_results,
        ordered_exp_ids,
        test_condition,
        "Training MSE Performance",
        "Epoch",
    )
else:
    print("No valid training MSE data found to plot.")

# Plot Validation MSE
if truncated_val_results:
    plot_mse_on_axis(
        ax_val,
        truncated_val_results,
        ordered_exp_ids,
        test_condition,
        "Validation MSE Performance",
        "Epoch",
    )
else:
    print("No valid validation MSE data found to plot.")

fig.tight_layout()
fig.subplots_adjust(wspace=0.3)  # Add gap between plots
plot_filename = "combined_mse_research_paper.png"
plot_path = os.path.join(output_dir, plot_filename)
plt.savefig(plot_path, dpi=300)
print(f"Saved combined MSE plot to: {plot_path}")
plt.close()

print("\nScript execution finished. Plots saved in:", output_dir)
