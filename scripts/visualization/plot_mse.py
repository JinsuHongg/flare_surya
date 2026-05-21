import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Configuration
BASE_DIR = "/home/jhong90/github_proj/flare_surya/results/train/light-weight_fm"
OUTPUT_DIR = "/home/jhong90/github_proj/flare_surya/results/plots"
MAX_EPOCH = 100

# Define data configurations
DATA_CONFIG = {
    "xrs": {"train": "xrs_train_mse.csv", "val": "xrs_val_mse.csv", "label": "XRS"},
    "halpha": {
        "train": "halpha_train_mse.csv",
        "val": "halpha_val_mse.csv",
        "label": "Halpha",
    },
    "c2": {"train": "c2_train_mse.csv", "val": "c2_val_mse.csv", "label": "C2"},
}

# Style mapping
STYLES = {
    "Small": {"linestyle": "-", "marker": "o", "color": "#0072B2"},
    "Medium": {"linestyle": "--", "marker": "s", "color": "#D55E00"},
    "Large": {"linestyle": ":", "marker": "^", "color": "#009E73"},
    "Base": {"linestyle": "--", "marker": "s", "color": "#D55E00"},
}
DEFAULT_STYLE = {"linestyle": "-", "marker": "o", "color": "#0072B2"}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_display_name(col):
    """Generates a display name from the column name."""
    base = col.split(" - ")[0].lower()
    if "small" in base or "test13" in base:
        return "Small"
    if "test14" in base:
        return "Medium"
    if "test12" in base:
        return "Large"
    if "halpha" in base:
        return "Base"
    if "c2" in base or "lasco" in base:
        return "Base"
    return base.replace("_", " ").title()


def extract_experiment_mse_data(df, mse_type):
    """Extracts MSE data for specified experiments from a DataFrame."""
    experiment_data = {}
    mse_col_pattern = re.compile(rf" - {re.escape(mse_type)}$")

    potential_mse_cols = [col for col in df.columns if mse_col_pattern.search(col)]

    for col in potential_mse_cols:
        display_name = get_display_name(col)
        mse_series = df[col].dropna()

        if not mse_series.empty:
            x_axis_col = "epoch" if "epoch" in df.columns else "trainer/global_step"
            if x_axis_col in df.columns:
                x_axis_data = df[x_axis_col].dropna()
                min_len = min(len(x_axis_data), len(mse_series))
                if min_len > 0:
                    experiment_data[display_name] = (
                        x_axis_data.iloc[:min_len],
                        mse_series.iloc[:min_len],
                    )

    return experiment_data


def plot_mse_on_axis(ax, results, title, xlabel):
    """Plots MSE data on a given axis with log scale and specific styles/font sizes."""
    for display_name, (x_axis, mse_series) in results.items():
        # Filter data
        mask = x_axis <= MAX_EPOCH
        x_axis_filtered = x_axis[mask]
        mse_series_filtered = mse_series[mask]

        # Get style
        style = STYLES.get(display_name, DEFAULT_STYLE)

        # Plot
        lines = ax.plot(
            x_axis_filtered,
            mse_series_filtered,
            label=display_name,
            linewidth=2.5,
            markersize=6,
            **style,
        )
        lines[0].set_markevery(max(1, len(x_axis_filtered) // 10))

    ax.set_title(title, fontsize=28, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=26)

    # Log scale settings
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)

    ax.legend(fontsize=24, frameon=True)
    ax.tick_params(axis="both", which="major", labelsize=24)


# Create 3x2 Plot
fig, axes = plt.subplots(3, 2, figsize=(24, 24))

for i, (data_type, config) in enumerate(DATA_CONFIG.items()):
    train_path = os.path.join(BASE_DIR, config["train"])
    val_path = os.path.join(BASE_DIR, config["val"])

    # Load data
    try:
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
    except FileNotFoundError as e:
        print(f"Error loading {data_type} data: {e}")
        continue

    # Extract data
    train_results = extract_experiment_mse_data(df_train, "train/mse")
    val_results = extract_experiment_mse_data(df_val, "val/mse")

    # Plot
    plot_mse_on_axis(
        axes[i, 0], train_results, f"{config['label']} Training Performance", "Epoch"
    )
    plot_mse_on_axis(
        axes[i, 1], val_results, f"{config['label']} Validation Performance", "Epoch"
    )

fig.tight_layout()
fig.subplots_adjust(wspace=0.3, hspace=0.3)

# Save PNG for preview
plt.savefig(
    os.path.join(OUTPUT_DIR, "combined_mse_analysis.png"), dpi=300, bbox_inches="tight"
)
# Save PDF for journal
pdf_path = os.path.join(OUTPUT_DIR, "combined_mse_analysis.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
print(f"Saved journal-quality PDF to: {pdf_path}")
plt.close()
