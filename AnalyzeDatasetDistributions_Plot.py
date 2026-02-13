import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
PLOT_STYLE = {
    # Fonts and Sizes
    'font_family': 'serif',       # 'serif' (Times), 'sans-serif' (Arial), etc.
    'row_title_size': 24,
    'title_size': 24,
    'top_title_size': 28,
    'axis_label_size': 20,
    'tick_label_size': 18,
    'cbar_label_size': 18,
    'cbar_offset_size': 16,   # Size for the 1e-4 text
    
    # Plotting Parameters
    'dpi': 1200,                # 300 for print, 1200 for high-res
    'bins': 250,                  # Number of bins for the histogram
    'cmap_density': 'jet',        # Colormap for standard density (jet, viridis, inferno)
    'cmap_diff': 'seismic',       # Diverging colormap for difference (seismic, coolwarm, bwr)
    
    # Axis Limits
    'gaze_lim': 120,              # Limit for Gaze plots (+/- deg)
    'head_lim': 80,               # Limit for Head plots (+/- deg)
    
    # Layout Padding
    'title_pad': 20,        # Padding between title and plot
    'layout_wspace': 0.6,   # Increased space between subplot columns
    'layout_hspace': 0.2
}

# Apply font family globally
plt.rcParams['font.family'] = PLOT_STYLE['font_family']

INPUT_FILE = 'distributions_data.pkl'

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_histogram(yaw, pitch, limit):
    """Computes the 2D normalized probability density."""
    if len(yaw) == 0:
        return None
    
    # Compute 2D histogram
    # density=True ensures the integral of the histogram is 1
    H, xedges, yedges = np.histogram2d(
        yaw, pitch, 
        bins=PLOT_STYLE['bins'], 
        range=[[-limit, limit], [-limit, limit]], 
        density=True
    )
    return H.T  # Transpose for correct plotting orientation (y-axis is rows)

def plot_heatmap(ax, H, limit, title, cmap, vmin=None, vmax=None, is_diff=False, show_title = True):
    """Plots a single heatmap on the provided axis."""
    if H is None:
        ax.axis('off') # Hides the box/spines entirely
        #ax.set_title(title, fontsize=PLOT_STYLE['title_size'])
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha='center', va='center', color="gray", style='italic',
                fontsize=PLOT_STYLE['title_size'])
        return None

    # Extent defines the data coordinates of the image area
    extent = [-limit, limit, -limit, limit]
    
    im = ax.imshow(H, interpolation='nearest', origin='lower', 
                   extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Styling
    if show_title:
        ax.set_title(title, fontsize=PLOT_STYLE['title_size'], pad = PLOT_STYLE['title_pad'])
        
    ax.set_xlabel(r"Yaw ($^\circ$)", fontsize=PLOT_STYLE['axis_label_size'])
    ax.set_ylabel(r"Pitch ($^\circ$)", fontsize=PLOT_STYLE['axis_label_size'])
    
    # Ticks
    ax.tick_params(axis='both', which='major', labelsize=PLOT_STYLE['tick_label_size'])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(limit / 2)) # Ticks at -Lim, -Lim/2, 0...
    ax.yaxis.set_major_locator(ticker.MultipleLocator(limit / 2))
    
    # Grid and Crosshairs
    ax.grid(True, alpha=0.2, linestyle='--')
    color_cross = 'black' if is_diff else 'white'
    ax.axhline(0, color=color_cross, alpha=0.3, linestyle=':', linewidth=1)
    ax.axvline(0, color=color_cross, alpha=0.3, linestyle=':', linewidth=1)
    
    return im

def plot_dataset_analysis(name, dataset_data, inversion=False):
    """
    Generates a 2x3 Grid:
    Row 1: Gaze (Original, Extracted, Difference)
    Row 2: Head (Original, Extracted, Difference)
    """
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=False,
                            gridspec_kw={'hspace':PLOT_STYLE['layout_hspace'],'wspace':PLOT_STYLE['layout_wspace']})

    #fig.set_constrained_layout_pads(wspace=PLOT_STYLE['layout_wspace'], h_pad = PLOT_STYLE['layout_hspace'])
    
    # Helper to extract data
    def get_data(key):
        y, p = dataset_data[key]
        if inversion and len(y) > 0:
            return -y, -p
        return y, p

    # 1. Load Data
    g_orig = get_data('orig_gaze')
    g_extr = get_data('extr_gaze')
    g_diff = get_data('diff_gaze')
    h_orig = get_data('orig_head')
    h_extr = get_data('extr_head')
    h_diff = get_data('diff_head')

    # 2. Compute Histograms
    lim_g = PLOT_STYLE['gaze_lim']
    lim_h = PLOT_STYLE['head_lim']

    Hg_orig = get_histogram(g_orig[0], g_orig[1], lim_g)
    Hg_extr = get_histogram(g_extr[0], g_extr[1], lim_g)
    Hg_diff = get_histogram(g_diff[0], g_diff[1], lim_g)
    Hh_orig = get_histogram(h_orig[0], h_orig[1], lim_h)
    Hh_extr = get_histogram(h_extr[0], h_extr[1], lim_h)
    Hh_diff = get_histogram(h_diff[0], h_diff[1], lim_h)

    # 4. Determine Scales (Vmin/Vmax)
    # For density, we find max value across orig/extr to share the scale
    max_dens_g = max(np.max(Hg_orig) if Hg_orig is not None else 0, np.max(Hg_extr) if Hg_extr is not None else 0)
    max_dens_h = max(np.max(Hh_orig) if Hh_orig is not None else 0, np.max(Hh_extr) if Hh_extr is not None else 0)

    # 5. Plotting
    # --- Row 1: Gaze ---
    im1 = plot_heatmap(axs[0,0], Hg_orig, lim_g, r"Original", PLOT_STYLE['cmap_density'], 0, max_dens_g)
    im2 = plot_heatmap(axs[0,1], Hg_extr, lim_g, r"Extracted", PLOT_STYLE['cmap_density'], 0, max_dens_g)
    im3 = plot_heatmap(axs[0,2], Hg_diff, lim_g, r"Excluded", PLOT_STYLE['cmap_diff'], vmin = 0, is_diff=True)
    
    # --- Row 2: Head ---
    im4 = plot_heatmap(axs[1,0], Hh_orig, lim_h, r"Original", PLOT_STYLE['cmap_density'], 0, max_dens_h, show_title = False)
    im5 = plot_heatmap(axs[1,1], Hh_extr, lim_h, r"Extracted", PLOT_STYLE['cmap_density'], 0, max_dens_h, show_title = False)
    im6 = plot_heatmap(axs[1,2], Hh_diff, lim_h, r"Excluded", PLOT_STYLE['cmap_diff'], vmin=0, is_diff=True, show_title = False)

    # 6. Colorbars
    def add_cbar(im, ax, pad=0.05): #, label=r"Density"):
        if im is None: return
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=pad)
        cbar.ax.tick_params(labelsize=PLOT_STYLE['tick_label_size'])
        cbar.formatter.set_powerlimits((0, 0)) # Force scientific notation
        #cbar.set_label(label, fontsize=PLOT_STYLE['cbar_label_size'])
        cbar.ax.yaxis.get_offset_text().set_fontsize(PLOT_STYLE['cbar_offset_size']) # Set font size for the scientific notation (1e-4) offset text

    add_cbar(im1, axs[0,0])
    add_cbar(im2, axs[0,1])
    add_cbar(im3, axs[0,2])
    add_cbar(im4, axs[1,0])
    add_cbar(im5, axs[1,1])
    add_cbar(im6, axs[1,2])

    fig.suptitle(f"{name}", fontsize=PLOT_STYLE['top_title_size'], fontweight='bold')

    # 7. Add Row Labels on the Right
    row_labels = ["Gaze", "Head"]
    for i, label in enumerate(row_labels):
        # Target the last axis in each row (the Difference plots)
        ax = axs[i, 0] 
        ax.annotate(label, xy=(-0.52, 0.5), xycoords='axes fraction',
                    rotation=90, va='center', ha='right',
                    fontsize=PLOT_STYLE['row_title_size'], 
                    fontweight='bold')
    
    # Save
    safe_name = name.replace(" ", "_")
    print(f"Saving plots for {name}...")
    plt.savefig(f"dataset_distributions/{safe_name}.png", dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.savefig(f"dataset_distributions/{safe_name}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"dataset_distributions/{safe_name}.svg", format='svg', bbox_inches='tight')
    plt.close()

    # Calculate Retention Stats
    n_orig = len(g_orig[0])
    n_extr = len(g_extr[0])
    retention = (n_extr / n_orig * 100) if n_orig > 0 else 0
    print(f"--- {name} Statistics ---")
    print(f"  Original Samples:  {n_orig}")
    print(f"  Extracted Samples: {n_extr}")
    print(f"  Retention Rate:    {retention:.2f}%")
    print("")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    try:
        with open(INPUT_FILE, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Error: Pickle file not found.")
        exit()

    if 'gaze360' in data:
        plot_dataset_analysis("Gaze360", data['gaze360'], inversion=True)

    if 'xgaze' in data:
        plot_dataset_analysis("ETH-XGaze", data['xgaze'])

    if 'gazegene' in data:
        plot_dataset_analysis("GazeGene", data['gazegene'])
        
    print("All plots generated.")
