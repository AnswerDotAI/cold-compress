import matplotlib.pyplot as plt


if __name__ == "__main__":
    sequence_lengths = [8192, 16384, 32768, 65536]
    baseline_tokens = [20.82, 19.13, 11.77, 6.60]
    compile_tokens = [69.61, 50.18, 30.03, 17.97]
    compress_tokens = [73.26, 71.94, 71.93, 71.81]

    baseline_memory = [1.04, 2.12, 4.24, 8.79]
    compile_memory = [1.04, 2.12, 4.24, 8.79]
    compress_memory = [0.52, 0.52, 0.52, 0.52]

    baseline_perplexity = [10.69, 9.53, 10.45, 10.52]
    compile_perplexity = [10.69, 10.63, 10.45, 10.52]
    compress_perplexity = [10.70, 9.69, 10.59, 10.70]

    # Set up the plot
    plt.figure(figsize=(20, 8))
    # Custom style with larger fonts, especially for axes
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 28,
        'axes.titlesize': 30,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 20,
        'figure.titlesize': 32,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        'axes.edgecolor': '#888888',
        'axes.linewidth': 1.5,
    })
    # plt.style.use('fivethirtyeight')

    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # First subplot: Tokens / Second
    plt.subplot(121)
    plt.plot(sequence_lengths, baseline_tokens, label='Baseline', color=colors[0], linewidth=5)
    plt.plot(sequence_lengths, compile_tokens, label='+ Compile', color=colors[1], linewidth=5)
    plt.plot(sequence_lengths, compress_tokens, label='+ Compile + Compression', color=colors[2], linewidth=5)
    plt.xlabel('Sequence Length', fontsize=22)
    plt.ylabel('Tokens / Second', fontsize=22)
    plt.title('Tokens / Second vs Sequence Length', fontsize=26)
    plt.legend(fontsize=20)

    # Second subplot: KV Cache Memory GB
    plt.subplot(122)
    plt.plot(sequence_lengths, baseline_memory, label='Baseline', color=colors[0], linewidth=5)
    plt.plot(sequence_lengths, compile_memory, label='+ Compile', color=colors[1], linewidth=5)
    plt.plot(sequence_lengths, compress_memory, label='+ Compile + Compression', color=colors[2], linewidth=5)
    plt.xlabel('Sequence Length', fontsize=22)
    plt.ylabel('KV Cache Memory (GB)', fontsize=22)
    plt.title('KV Cache Memory vs Sequence Length', fontsize=26)
    plt.legend(fontsize=20)

    # Uncomment to add the third plot and change plt.subplot numbers above to 131 and 132
    # Third subplot: Perplexity
    # plt.subplot(133)
    # plt.plot(sequence_lengths, baseline_perplexity, label='Baseline', color=colors[0], linewidth=3)
    # plt.plot(sequence_lengths, compile_perplexity, label='+ Compile', color=colors[1], linewidth=3)
    # plt.plot(sequence_lengths, compress_perplexity, label='+ Compile + Compression', color=colors[2], linewidth=3)
    # plt.xlabel('Sequence Length', fontsize=22)
    # plt.ylabel('Perplexity', fontsize=22)
    # plt.title('Perplexity vs Sequence Length', fontsize=26)
    # plt.legend(fontsize=20)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('performance_graphs.png', dpi=300, bbox_inches='tight')
