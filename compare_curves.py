#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import argparse
import os


def load_history(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_histories(files):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ----- Plot Loss Curves -----
    for file in files:
        data = load_history(file)
        label = data.get("model_name", os.path.basename(file))
        axes[0].plot(data["train_loss"], label=f"{label} - train")
        axes[0].plot(data["val_loss"], linestyle="--", label=f"{label} - val")

    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # ----- Plot Accuracy Curves -----
    for file in files:
        data = load_history(file)
        label = data.get("model_name", os.path.basename(file))
        axes[1].plot(data["train_acc"], label=f"{label} - train")
        axes[1].plot(data["val_acc"], linestyle="--", label=f"{label} - val")

    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare learning curves from multiple model history files."
    )
    parser.add_argument(
        "--histories",
        nargs="+",
        required=True,
        help="List of history JSON files to compare."
    )
    args = parser.parse_args()

    # Validate that at least one file was provided
    if len(args.histories) == 0:
        raise ValueError("No history files were provided. Please specify at least one JSON file.")

    # Validate file names and existence
    for file in args.histories:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: '{file}'. Please check the path and try again.")
        if not (os.path.basename(file).startswith("history") and file.endswith(".json")):
            raise ValueError(
                f"Invalid file name: '{file}'. "
                "Expected a file starting with 'history' and ending with '.json'."
            )

    plot_histories(args.histories)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
