import os
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict

# --- Configuration (保持不变) ---
BASE_LOG_DIR = Path("/home/aiops/zhangfz/Muon_linear_regression/Muon_ICL/logs/linear_regression_tail/muon_lr_search")
MODEL_PARAM = "qkvo"
SEEDS = [42]#range(42, 44)
#MODES = [5] #range(0, 9)  # 扩展到包括 mode 6, 7, 8
VAL_LOSS_EVERY = 100  # Validation loss is logged every 100 steps
MAX_STEPS_TO_PLOT = 10000
# adam_lr_list = [2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4]
# adam_lr_list = [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]
# MODES = [str(adam_lr) for adam_lr in adam_lr_list]
muon_lr_list = [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]
# muon_lr_list = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
# MODES = [str(adam_lr) for adam_lr in adam_lr_list]
MODES = [str(muon_lr) for muon_lr in muon_lr_list]

MODE_LABELS = {
    0: "Mode 0 (Muon: All Attn+MLP)",
    1: "Mode 1 (Muon: QK Attn)",
    2: "Mode 2 (Muon: VO Attn)",
    3: "Mode 3 (Muon: All Attn)",
    4: "Mode 4 (Muon: MLP)",
    5: "Mode 5 (Adam: All Attn+MLP)",
    6: "Mode 6  Muon(W_2 MLP)/Adam(attn, W_1 MLP)",
    7: "Mode 7  Muon(VO Attn, MLP)/Adam(QK Attn)",
    8: "Mode 8  Muon(VO Attn, W_2 MLP)/Adam(QK Attn, W_1 MLP)",
}

color_map = plt.colormaps.get_cmap('tab10')
colors = [color_map(i) for i in np.linspace(0, 1, 10)]
mode_colors = [colors[i] for i in [0, 1, 2, 3, 4, 5, 6, 8, 9]]
markers = ['.', ',', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']

# Patterns to match train loss and val loss
train_loss_pattern = re.compile(
    r"train loss at step (\d+):\s*([\d.]+)"
)
val_loss_pattern = re.compile(
    r"val loss at step (\d+):\s*\[([\d\.,\s\-eE\+]+)\]"
)


# --- Data Aggregation ---
# Separate data structures for train loss and val loss
# For val loss, we store three separate arrays (one for each value in the array)
train_loss_data = defaultdict(lambda: defaultdict(lambda: {'losses': []}))
val_loss_data = defaultdict(lambda: defaultdict(lambda: {'losses_0': [], 'losses_1': [], 'losses_2': []}))

for seed in SEEDS:
    for mode in MODES:
        # run_folder_name = f"mode_adam_adam_lr_{mode}_seed_{seed}"
        run_folder_name = f"mode_muon_muon_lr_{mode}_seed_{seed}"
        # run_folder_name = f"mode_0_param_{MODEL_PARAM}_muon_lr_{mode}_seed_{seed}"
        run_dir_path = BASE_LOG_DIR / run_folder_name
        if not run_dir_path.is_dir():
            print(f"Warning: Directory not found for seed {seed}, mode {mode}: {run_dir_path}")
            continue
        log_files = list(run_dir_path.glob("log.txt"))
        if not log_files:
            print(f"Warning: No log.txt found in {run_dir_path}")
            continue
        log_file_path = log_files[0]
        print(f"Processing: {log_file_path}")
        try:
            with open(log_file_path, 'r') as f:
                for line in f:
                    # Parse train loss
                    train_match = train_loss_pattern.search(line)
                    if train_match:
                        step = int(train_match.group(1))
                        train_loss = float(train_match.group(2))
                        if step == 0:
                            continue
                        if step > MAX_STEPS_TO_PLOT:
                            continue
                        train_loss_data[mode][step]['losses'].append(train_loss)
                    
                    # Parse val loss (array of 3 values)
                    val_match = val_loss_pattern.search(line)
                    if val_match:
                        step = int(val_match.group(1))
                        # Parse the list of losses: "[value1, value2, value3]"
                        loss_str = val_match.group(2)
                        # Split by comma and convert to floats
                        loss_values = [float(x.strip()) for x in loss_str.split(',') if x.strip()]
                        if len(loss_values) < 3:
                            continue
                        if step == 0:
                            continue
                        if step > MAX_STEPS_TO_PLOT:
                            continue
                        # Store each of the three loss values separately
                        val_loss_data[mode][step]['losses_0'].append(loss_values[0])
                        val_loss_data[mode][step]['losses_1'].append(loss_values[1])
                        val_loss_data[mode][step]['losses_2'].append(loss_values[2])
        except Exception as e:
            print(f"Error processing {log_file_path}: {e}")

# --- Calculate Averages ---
# For train loss
train_loss_results = defaultdict(lambda: {'steps': [], 'avg_losses': []})
# For val loss - three separate results (one for each value in the array)
val_loss_results = defaultdict(lambda: {
    'steps': [], 
    'avg_losses_0': [], 
    'avg_losses_1': [], 
    'avg_losses_2': []
})

common_plot_steps = sorted(list(range(VAL_LOSS_EVERY, MAX_STEPS_TO_PLOT + 1, VAL_LOSS_EVERY)))

# Process train loss
for mode in MODES:
    for step in common_plot_steps:
        if step in train_loss_data[mode]:
            data_at_step = train_loss_data[mode][step]
            if data_at_step['losses']:
                avg_loss = np.mean(data_at_step['losses'])
                train_loss_results[mode]['steps'].append(step)
                train_loss_results[mode]['avg_losses'].append(avg_loss)

# Process val loss (three separate values)
for mode in MODES:
    for step in common_plot_steps:
        if step in val_loss_data[mode]:
            data_at_step = val_loss_data[mode][step]
            if data_at_step['losses_0'] and data_at_step['losses_1'] and data_at_step['losses_2']:
                avg_loss_0 = np.mean(data_at_step['losses_0'])
                avg_loss_1 = np.mean(data_at_step['losses_1'])
                avg_loss_2 = np.mean(data_at_step['losses_2'])
                val_loss_results[mode]['steps'].append(step)
                val_loss_results[mode]['avg_losses_0'].append(avg_loss_0)
                val_loss_results[mode]['avg_losses_1'].append(avg_loss_1)
                val_loss_results[mode]['avg_losses_2'].append(avg_loss_2)

# --- Plotting ---
output_plot_dir = BASE_LOG_DIR
output_plot_dir.mkdir(parents=True, exist_ok=True)
print(f"Plots will be saved to: {output_plot_dir.resolve()}")

# --- Figure 1: Train Loss vs. Steps ---
plt.figure(figsize=(12, 7))
for i, mode in enumerate(sorted(train_loss_results.keys())):
    data = train_loss_results[mode]
    if data['steps'] and data['avg_losses']:
        plt.plot(
            data['steps'], data['avg_losses'],
            marker=markers[i % len(markers)], markersize=4, linestyle='-',
            linewidth=1.5, color=mode_colors[i % len(mode_colors)],
            label=f"LR {mode}"
        )
plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("Average Training Loss", fontsize=14)
plt.title("Training Loss vs. Steps (by Learning Rate)", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='upper right', fontsize='medium')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
save_path_train = output_plot_dir / "train_loss_vs_steps.png"
plt.savefig(save_path_train)
print(f"Saved plot: {save_path_train}")
plt.close()

# --- Figure 2: Val Loss vs. Steps (Three Lines) ---
plt.figure(figsize=(12, 7))
# Use different line styles for the three val loss values
line_styles = ['-', '--', '-.']
line_labels = ['Component [0]', 'Component [1]', 'Component [2]']

for i, mode in enumerate(sorted(val_loss_results.keys())):
    data = val_loss_results[mode]
    if data['steps']:
        # Plot three lines for each learning rate
        for loss_idx in range(3):
            loss_key = f'avg_losses_{loss_idx}'
            if data[loss_key]:
                plt.plot(
                    data['steps'], data[loss_key],
                    marker=markers[i % len(markers)], 
                    markersize=3, 
                    linestyle=line_styles[loss_idx],
                    linewidth=1.5, 
                    color=mode_colors[i % len(mode_colors)],
                    alpha=0.8 if loss_idx == 0 else 0.6,
                    label=f"LR {mode} - {line_labels[loss_idx]}"
                )

plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("Average Validation Loss", fontsize=14)
plt.title("Validation Loss vs. Steps (Three Components per Learning Rate)", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='upper right', fontsize='x-small', ncol=3)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
save_path_val = output_plot_dir / "val_loss_vs_steps.png"
plt.savefig(save_path_val)
print(f"Saved plot: {save_path_val}")
plt.close()

print("All plots generated.")
