import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# Import from training script
from Train_siameseMLP import Config, SiameseGazeDataset, SiameseGazeNet, GazeAngularLoss

# --- Configuration for Publication Plot ---
sns.set_context("paper", font_scale=1.5)  # larger fonts for readability
sns.set_style("whitegrid")  # Clean background
plt.rcParams['font.family'] = 'serif'

def get_grouped_indices(config):
    """Maps logical groups to tensor indices"""
    def map_indices(full_list, sub_list, offset=0):
        indices = []
        for i, lm_idx in enumerate(full_list):
            if lm_idx in sub_list:
                indices.extend([2*i + offset, 2*i + offset + 1])
        return indices

    groups = {}
    groups['Left iris'] = (0, map_indices(config.LEFT_EYE_LANDMARKS, config.LEFT_IRIS))
    groups['Right iris'] = (1, map_indices(config.RIGHT_EYE_LANDMARKS, config.RIGHT_IRIS))
    groups['Left contour'] = (0, map_indices(config.LEFT_EYE_LANDMARKS, config.LEFT_EYE_CONTOUR))
    groups['Right contour'] = (1, map_indices(config.RIGHT_EYE_LANDMARKS, config.RIGHT_EYE_CONTOUR))
    groups['Head anchors'] = (3, list(range(len(config.HEAD_ANCHORS) * 2)))
    groups['Relative pos.'] = (2, [0, 1])
    return groups

def evaluate_loss(model, inputs, targets, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(*inputs)
        loss = criterion(preds, targets)
    return loss.item()

def main(VALID_FILE,MODEL_SAVE_PATH,MODEL_NAME,plot_type):
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Data & Model
    valid_df = pd.read_csv(VALID_FILE, sep=';')
    valid_dataset = SiameseGazeDataset(valid_df, config, is_train=False)
    val_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    X_l, X_r, X_rel, X_h, y = next(iter(val_loader))
    inputs = [X_l.to(device), X_r.to(device), X_rel.to(device), X_h.to(device)]
    y = y.to(device)

    model = SiameseGazeNet(len(config.LEFT_EYE_LANDMARKS) * 2, config).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    criterion = GazeAngularLoss()

    # Baseline
    baseline = evaluate_loss(model, inputs, y, criterion)
    feature_groups = get_grouped_indices(config)
    
    # --- Run Permutation N times for Error Bars ---
    n_repeats = 1000
    data_records = []

    print(f"Running Permutation Importance ({n_repeats} repeats)...")
    
    for group_name, (tensor_idx, indices) in feature_groups.items():
        for i in range(n_repeats):
            # Clone and Corrupt
            temp_inputs = [t.clone() for t in inputs]
            target_tensor = temp_inputs[tensor_idx]
            
            # Shuffle specific indices
            perm = torch.randperm(target_tensor.size(0))
            # We select all rows, but only the specific feature columns
            corrupted_slice = target_tensor[:, indices][perm]
            target_tensor[:, indices] = corrupted_slice
            
            # Record Difference
            new_loss = evaluate_loss(model, temp_inputs, y, criterion)
            importance = new_loss - baseline
            
            data_records.append({
                'Feature group': group_name,
                'Importance': importance
            })

    df = pd.DataFrame(data_records)

    # --- Plotting ---
    plt.figure(figsize=(8, 5))

    if plot_type == "barplot":
        # Create the bar plot with error bars (ci='sd' shows Standard Deviation)
        ax = sns.barplot(
            data=df, 
            x='Importance', 
            y='Feature group',
            errorbar='sd',  # Standard Deviation bars
            capsize=.2,     # Caps on error bars
            color=".4",     # Grey scale
            edgecolor="black"
        )
    else:
        # 1. Draw Boxplot
        sns.boxplot(
            data=df, x='Importance', y='Feature group',
            hue='Feature group',legend=False,
            whis=[0, 100], width=.6, palette="vlag", linewidth=1.5, fliersize=0
        )
        
        # 2. Add Stripplot (Individual points)
        #sns.stripplot(
        #    data=df, x='Importance', y='Feature group',
        #    size=4, color=".3", linewidth=0, alpha=0.6
        #)

    # Formatting
    plt.axvline(0, color='black', linewidth=1, linestyle='--')
    plt.title(f'{MODEL_NAME}') 
    plt.xlabel(r'Increase in Mean Angular Error ($^\circ$)') #, fontweight='bold')
    plt.ylabel('')
    
    # Remove top and right spines (chartjunk)
    sns.despine()
    
    # Save high-res
    plt.tight_layout()
    plt.savefig(f'models/stats/PFI_siamese_{MODEL_NAME}_{plot_type}.png', dpi=1200, bbox_inches='tight')
    plt.savefig(f'models/stats/PFI_siamese_{MODEL_NAME}_{plot_type}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'models/stats/PFI_siamese_{MODEL_NAME}_{plot_type}.svg', format='svg', bbox_inches='tight')

    #plt.show()
    plt.close()

if __name__ == "__main__":

    plot_type = "boxplot"
    LIST = [
        {
            'name': '(c) ETH-Xgaze',
            'val': 'datasets/XGaze_448/xgaze_normalized_det_conf_0_8_VALID.csv',
            'model': 'models/xgaze_siameseMLP.pth',
        },{
            'name': '(b) GazeGene',
            'val': 'datasets/GazeGene/xgaze_normalized_det_conf_0_8_VALID.csv',
            'model': 'models/gazegene_siameseMLP.pth',
        },{
            'name': '(a) Gaze360',
            'val': 'datasets/Gaze360/gaze360_normalized_VAL.csv',
            'model': 'models/gaze360_siameseMLP.pth',
        }
    ]

    for item in LIST:
        main(item['val'], item['model'], item['name'],plot_type)
