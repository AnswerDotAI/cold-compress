import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    # Define the data
    df = pd.read_csv("/workspace/attention_loss.csv")

    decoding_steps = np.arange(500, 8500, 500)
    n = len(decoding_steps)
    models = ['Low Compression', 'Medium Compression', 'High Compression']

    # Sample data - replace with your actual data
    attention_loss = {
        'Low Compression': df["25_attention_loss"][:n],
        'Medium Compression': df["50_attention_loss"][:n],
        'High Compression': df["75_attention_loss"][:n],
    }

    ppl_delta = {
        'Low Compression': df["25_ppl_delta"][:n],
        'Medium Compression': df["50_ppl_delta"][:n],
        'High Compression': df["75_ppl_delta"][:n],
    }

    # Create the plot
    plt.rcParams.update({'font.size': 20})

    fig, ax1 = plt.subplots(figsize=(20, 10))

    # Colors for each model
    colors = ["#006AA7", '#16a085', '#8e44ad', '#d35400']

    # Plot Attention Loss
    for model, color in zip(models, colors):
        ax1.plot(decoding_steps, attention_loss[model], color=color, label=f'{model} (Attention Loss)', linewidth=6)
        ax1.scatter(decoding_steps, attention_loss[model], color=color, s=400)

    ax1.set_xlabel('Decoding Steps', fontsize=32)
    ax1.set_ylabel('Attention Loss', fontsize=32)

    ax1.tick_params(axis='y', labelsize=32)
    ax1.tick_params(axis='x', labelsize=32)

    # Create a second y-axis for PPL
    ax2 = ax1.twinx()

    # Plot Perplexity (PPL)
    for model, color in zip(models, colors):
        ax2.plot(decoding_steps, ppl_delta[model], color=color, linestyle='--', label=f'{model} (PPL Δ)', linewidth=6)
        ax2.scatter(decoding_steps, ppl_delta[model], color=color, marker='s', s=400)

    ax2.set_ylabel("Perplexity Delta (PPL Δ)", fontsize=32)
    ax2.tick_params(axis="y", labelsize=32)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.05, 0.95), borderaxespad=0.25, fontsize=24)

    plt.title("Attention Loss & Perplexity vs Decoding Steps", fontsize=32)
    plt.grid(True)
    plt.tight_layout()
    # Save a plot to ../images directory
    # Get the current directory
    current_dir = Path(__file__).resolve().parent
    # Save the plot to the desired path
    plt.savefig(current_dir.parent / "images" / "attention_loss_pg19.png")
