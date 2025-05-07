import matplotlib.pyplot as plt
import pandas as pd

def load_accuracy_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                try:
                    epoch, acc = line.strip().split(':')
                    data.append((int(epoch.strip()), float(acc.strip())))
                except ValueError:
                    continue  # Skip malformed lines
    return pd.DataFrame(data, columns=['Epoch', 'Accuracy'])

# Replace with your actual paths
foveated_file = 'fine_tuning_foveated_output.txt'
unfoveated_file = 'fine_tuning_unfoveated_output.txt'

# Load data
foveated_df = load_accuracy_file(foveated_file)
unfoveated_df = load_accuracy_file(unfoveated_file)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(foveated_df['Epoch'], foveated_df['Accuracy'], label='Foveated', marker='o')
plt.plot(unfoveated_df['Epoch'], unfoveated_df['Accuracy'], label='Unfoveated', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()