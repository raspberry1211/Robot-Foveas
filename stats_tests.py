import re
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

import pandas as pd

def extract_metrics(filepath):
    inference_times = []
    accuracies = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines) - 1:
        line_time = lines[i].strip()
        line_acc = lines[i + 1].strip()

        # Skip if either line is an average
        if "average" in line_time.lower() or "average" in line_acc.lower():
            i += 1
            continue

        try:
            time = float(line_time.split(":")[1].strip().split()[0])
            acc = float(line_acc.split(":")[1].strip().strip('%'))
            inference_times.append(time)
            accuracies.append(acc)
            i += 2  # Only increment if both lines were valid
        except Exception as e:
            print(f"Skipping lines {i}/{i+1} due to error: {e}")
            i += 1  # Move forward in case of error

    return pd.DataFrame({
        'inference_time': inference_times,
        'accuracy': accuracies
    })

# Load both logs
foveated_df = extract_metrics('fovea_race_output.txt')
unfoveated_df = extract_metrics('unfovea_race_output.txt')

print("Foveated Data:")
print(foveated_df.head(), "\nTotal rows:", len(foveated_df))

print("Unfoveated Data:")
print(unfoveated_df.head(), "\nTotal rows:", len(unfoveated_df))

# Summary stats
print("\n=== Summary Statistics ===")
for name, df in [('Foveated', foveated_df), ('Unfoveated', unfoveated_df)]:
    print(f"\n{name}:")
    print(df.describe())

# T-tests
print("\n=== T-Tests ===")
time_ttest = stats.ttest_ind(foveated_df['inference_time'], unfoveated_df['inference_time'], equal_var=False)
acc_ttest = stats.ttest_ind(foveated_df['accuracy'], unfoveated_df['accuracy'], equal_var=False)

print(f"Inference Time t-test: t={time_ttest.statistic:.3f}, p={time_ttest.pvalue:.4f}")
print(f"Accuracy t-test:       t={acc_ttest.statistic:.3f}, p={acc_ttest.pvalue:.4f}")

# Optional: Plot histograms
plot = True
if plot:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(foveated_df['inference_time'], bins=20, alpha=0.7, label='Foveated')
    plt.hist(unfoveated_df['inference_time'], bins=20, alpha=0.7, label='Unfoveated')
    plt.title('Inference Time Distribution')
    plt.xlabel('Seconds')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(foveated_df['accuracy'], bins=10, alpha=0.7, label='Foveated')
    plt.hist(unfoveated_df['accuracy'], bins=10, alpha=0.7, label='Unfoveated')
    plt.title('Accuracy Distribution')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()