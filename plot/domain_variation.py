import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

# Configure font for clarity
rcParams.update({'font.size': 10})

# Raw metric data: [model][domain]
mel = np.array([
    [0.523, 0.546, 0.638],
    [1.531, 1.472, 1.233],
    [0.778, 0.829, 0.945],
    [0.585, 0.616, 0.708]
])

stft = np.array([
    [1.314, 1.438, 1.826],
    [6.064, 5.553, 4.606],
    [1.503, 1.805, 2.246],
    [1.362, 1.554, 1.957]
])

wf = np.array([
    [0.011, 0.014, 0.033],
    [0.033, 0.028, 0.067],
    [0.026, 0.025, 0.062],
    [0.014, 0.016, 0.038]
])

sisdr = np.array([
    [13.156, 11.496, 7.31],
    [0.589, 2.059, 4.082],
    [4.872, 4.855, -0.435],
    [10.756, 9.431, 5.743]
])

fad = np.array([
    [0.183, 0.257, 0.820],
    [4.453, 2.964, 4.321],
    [1.000, 1.489, 2.428],
    [0.465, 0.518, 0.948]
])

metrics = {
    'Mel': mel,
    'STFT': stft,
    'Waveform': wf,
    'SI-SDR': sisdr,
    'FAD': fad
}

model_names = ['DAC', 'SAT', 'SNAC', 'Proposed']

# Calculate domain-wise variation (max - min) for each model in each metric
variation_data = {
    metric_name: values.max(axis=1) - values.min(axis=1)
    for metric_name, values in metrics.items()
}

# Normalize each metric's variation to [0, 1] range for fair comparison across metrics
normalized_variation_data = {
    metric_name: (variations - variations.min()) / (variations.max() - variations.min())
    for metric_name, variations in variation_data.items()
}

# Plot normalized variation
fig, ax = plt.subplots(figsize=(10, 5))

# Plot all variations in a single grouped bar chart
x = np.arange(len(model_names))  # 4 models
width = 0.15
offsets = np.linspace(-2, 2, len(variation_data)) * width

for i, (metric_name, variations) in enumerate(normalized_variation_data.items()):
    ax.bar(x + offsets[i], variations, width=width, label=metric_name)

ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylabel('Normalized Domain-wise Variation')
ax.legend()
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

plt.tight_layout()
norm_path = "normalized_domain_variation_barplot.png"
plt.savefig(norm_path)
plt.close()