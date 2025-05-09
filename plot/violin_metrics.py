import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

models = ['dac_9', 'sat', 'snac', 'uwavescale_16']
labels = ['DAC', 'SAT', 'SNAC', 'Proposed']
domains = ['speech', 'music', 'environment']
metrics = ['mel', 'stft', 'waveform', 'sisdr']
sr = '44k'

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, metric in zip(axes, metrics):
    index = f'{metric}-{sr}'
    records = []

    for model, label in zip(models, labels):
        for domain in domains:
            path = f"results/decode/{model}/{domain}/metrics.csv"
            try:
                values = pd.read_csv(path)[index]
                for v in values:
                    records.append({
                        'Model': label,
                        'Domain': domain,
                        'Value': v
                    })
            except Exception as e:
                print(f"Failed to read {path}: {e}")

    df_all = pd.DataFrame(records)

    sns.violinplot(x='Model', y='Value', hue='Domain', data=df_all,
                   inner='box', palette='Set2', ax=ax)

    ax.set_title(f"{metric.upper()} - {sr}")
    ax.set_xlabel("")
    ax.set_ylabel("")

plt.tight_layout()
plt.savefig("plot/violin/combined_violin_metrics_highres.png", dpi=300)
plt.close()
