import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress pandas warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# 1. DATA INGESTION & CLEANING

def load_ncu_csv(filepath):
    print(f"[1.] Loading and cleaning NCU data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find {filepath}. Ensure the file is in the same directory.")

    # Find the row where the actual CSV header starts ("ID")
    header_idx = 0
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('"ID"'):
                header_idx = i
                break
                
    # Read CSV starting from the header
    df = pd.read_csv(filepath, skiprows=header_idx)
    
    # Drop the first row which contains units (e.g., '%', 'byte') instead of data
    if pd.isna(df.iloc[0]['ID']):
        df = df.drop(index=0).reset_index(drop=True)
        
    # Convert numerical strings (like "1,024") to floats
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(',', '').astype(float)
            except ValueError:
                pass # Keep as string if it's actual text (like Kernel Name)
                
    return df

# 2. METRICS EXTRACTION

def extract_hardware_metrics(df, kernel_index=0):
    """
    Extracts Speed of Light (SOL), Occupancy, and Warp Stall metrics for a specific kernel invocation.
    """
    
    print("[3.] Extracting microarchitectural metrics...")
    kernel_data = df.iloc[kernel_index]
    kernel_name = kernel_data.get('Kernel Name', 'Unknown Kernel')
    
    # A. Speed of Light (SOL) Metrics
    sol_metrics = {
        'Compute (SM)': kernel_data.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 0.0),
        'Memory (DRAM)': kernel_data.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 0.0)
    }
    
    # B. Warp Stall Reasons (Filtering for metrics ending in '.pct')
    stall_cols = [c for c in df.columns if 'smsp__warp_issue_stalled' in c and c.endswith('.pct')]
    stall_metrics = {}
    for col in stall_cols:
        val = kernel_data.get(col, 0.0)
        # Filter out negligible stalls (< 1%) to keep visualizations clean
        if pd.notnull(val) and val > 1.0:
            # Clean up the metric name (e.g., 'smsp__warp_issue_stalled_imc_miss_per_warp_active.pct' -> 'Imc Miss')
            clean_name = col.split('stalled_')[-1].replace('_per_warp_active.pct', '').replace('_', ' ').title()
            stall_metrics[clean_name] = val
            
    # C. Occupancy Limiters
    occupancy_limits = {
        'Warps': kernel_data.get('launch__occupancy_limit_warps', 0),
        'Registers': kernel_data.get('launch__occupancy_limit_registers', 0),
        'Shared Mem': kernel_data.get('launch__occupancy_limit_shared_mem', 0),
        'Blocks': kernel_data.get('launch__occupancy_limit_blocks', 0)
    }
    
    return kernel_name, sol_metrics, stall_metrics, occupancy_limits

# 3. AUTOMATED DATA VISUALIZATION

def generate_hardware_dashboard(kernel_name, sol, stalls, occupancy):
    """
    Generates a professional 2x2 dashboard visualizing GPU bottlenecks,
    styled for data science and analytics reporting.
    """
    
    print("[4.] Generating hardware analysis dashboard...")
    
    # Set Seaborn theme and scaling for a professional report look
    sns.set_theme(style="whitegrid", context="talk")
    
    # Set up a 2x2 grid for plots with a larger figure size
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Modern, sleek title
    fig.suptitle(f'Hardware Profiling Report | Kernel: {kernel_name}', 
                 fontsize=22, fontweight='bold', y=0.96, color='#2c3e50')
    
    # ==========================================
    # --- Plot 1: Speed of Light (Top Left) ---
    # ==========================================
    ax1 = axes[0, 0]
    sns.barplot(x=list(sol.keys()), y=list(sol.values()), ax=ax1, palette="mako")
    ax1.set_title('Speed of Light (SOL) Utilization', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('% of Peak Hardware Capacity', fontsize=14)
    ax1.set_ylim(0, 100)
    
    # Annotate bars with values
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.1f}%", 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 10), textcoords='offset points', 
                     fontweight='bold', fontsize=13)

    # ==========================================
    # --- Plot 2: Warp Stall Distribution (Top Right) ---
    # ==========================================
    ax2 = axes[0, 1]
    if stalls:
        # Sort stalls by value (highest to lowest) for better data readability
        sorted_stalls = dict(sorted(stalls.items(), key=lambda item: item[1], reverse=True))
        
        # Horizontal bar chart is much better for categorical data than pie charts
        sns.barplot(x=list(sorted_stalls.values()), y=list(sorted_stalls.keys()), ax=ax2, palette="flare")
        ax2.set_title('Warp Stall Distribution', fontsize=16, fontweight='bold', pad=15)
        ax2.set_xlabel('% of Active Warps Stalled', fontsize=14)
        
        # Annotate bars on the right
        for p in ax2.patches:
            width = p.get_width()
            ax2.annotate(f"{width:.1f}%", 
                         (width, p.get_y() + p.get_height() / 2.), 
                         ha='left', va='center', xytext=(8, 0), textcoords='offset points', 
                         fontsize=13)
    else:
        ax2.text(0.5, 0.5, 'No Significant Stalls Detected', ha='center', va='center', fontsize=14, color='gray')
        ax2.axis('off')

    # ==========================================
    # --- Plot 3: Occupancy Limiters (Bottom Left) ---
    # ==========================================
    ax3 = axes[1, 0]
    sns.barplot(x=list(occupancy.keys()), y=list(occupancy.values()), ax=ax3, palette="crest")
    ax3.set_title('Occupancy Limiters (Max Blocks / SM)', fontsize=16, fontweight='bold', pad=15)
    ax3.set_ylabel('Limit Count', fontsize=14)
    
    # Annotate integer limits
    for p in ax3.patches:
        ax3.annotate(f"{int(p.get_height())}", 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 10), textcoords='offset points', 
                     fontweight='bold', fontsize=13)
    
    # ==========================================
    # --- Plot 4: Automated Diagnosis (Bottom Right) ---
    # ==========================================
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Determine primary bottleneck
    is_compute_bound = sol['Compute (SM)'] > sol['Memory (DRAM)']
    primary_bound = "Compute Bound" if is_compute_bound else "Memory Bound"
    primary_stall = max(stalls, key=stalls.get) if stalls else "None"
    
    # Formulate report text
    diagnosis_text = (
        f"Automated Hardware Diagnosis\n\n"
        f"• Primary Bottleneck: {primary_bound}\n"
        f"• Peak Subsystem Utilization: {max(sol.values()):.1f}%\n"
        f"• Dominant Warp Stall: {primary_stall}\n\n"
        f"Strategic Recommendation:\n"
    )
    
    if is_compute_bound:
        diagnosis_text += "Focus on optimizing ALU operations, loop unrolling,\nand minimizing branch divergence."
    else:
        diagnosis_text += "Focus on memory coalescing, maximizing Shared\nMemory hits, and increasing arithmetic intensity."
        
    # Create a sleek, modern text box
    bbox_props = dict(boxstyle="round,pad=1.5", fc="#f8f9fa", ec="#ced4da", lw=2)
    ax4.text(0.05, 0.5, diagnosis_text, fontsize=15, verticalalignment='center',
             bbox=bbox_props, color='#343a40', linespacing=1.7)

    # ==========================================
    # --- Layout and Save ---
    # ==========================================
    # Adjust layout to prevent clipping, leaving room for the main title
    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 
    
    output_filename = 'gpu_architecture_report.png'
    # bbox_inches='tight' ensures the saved image has clean margins
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[5.] Report successfully saved to {output_filename}\n\n")
    plt.show()

# 4. MAIN EXECUTION ENTRY POINT

if __name__ == "__main__":
    csv_file = "../CUDA_KERNELS-main/ncu_metrics.csv"
    
    try:
        # Execute pipeline
        ncu_df = load_ncu_csv(csv_file)
        print(f"[2.] Successfully loaded {ncu_df.shape[0]} kernel records with {ncu_df.shape[1]} metrics.")
        
        # Analyze the first kernel in the CSV
        k_name, k_sol, k_stalls, k_occ = extract_hardware_metrics(ncu_df, kernel_index=0)
        
        # Visualize
        generate_hardware_dashboard(k_name, k_sol, k_stalls, k_occ)
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")