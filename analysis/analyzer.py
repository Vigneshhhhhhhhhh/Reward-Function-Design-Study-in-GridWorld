import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from config import RESULTS_DIR

def run_analysis(export_latex=True):
    """
    Parses the intermediate JSON telemetry and compiles summary statistics
    and findings. Generates a CSV table, LaTeX table code, and a text
    report containing key insights and Wilcoxon signed-rank test results.
    
    Parameters
    ----------
    export_latex : bool, optional
        Whether to export the summary statistics to a LaTeX formatted txt file.
        Defaults to True.
    """
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    if not os.path.exists(summary_path):
        print("Summary JSON not found, skipping analysis.")
        return
        
    with open(summary_path, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # Structure dataframe for analysis
    df['grid_size_str'] = df['grid_size'].apply(lambda x: f"{x[0]}x{x[1]}")
    
    # 1. Summary Statistics Table
    stats = []
    
    for size in df['grid_size_str'].unique():
        size_df = df[df['grid_size_str'] == size]
        
        for cond in size_df['condition'].unique():
            cond_df = size_df[size_df['condition'] == cond]
            
            # Handle successful vs failed/timeout runs
            success_df = cond_df[cond_df['status'] == 'success']
            
            mean_iters = success_df['iterations_to_converge'].mean() if not success_df.empty else float('nan')
            std_iters = success_df['iterations_to_converge'].std() if not success_df.empty else float('nan')
            opt_score = success_df['optimality_score'].mean() if not success_df.empty else float('nan')
            avg_time = success_df['convergence_time'].mean() if not success_df.empty else float('nan')
            
            # Policy Quality: successful convergence %
            policy_quality = len(success_df) / len(cond_df) * 100 if len(cond_df) > 0 else 0
            
            stats.append({
                "Grid Size": size,
                "Reward Condition": cond,
                "Mean Convergence Iters": mean_iters,
                "Std": std_iters,
                "Optimality Score": opt_score,
                "Policy Quality (%)": policy_quality,
                "Avg Time(s)": avg_time
            })
            
    stats_df = pd.DataFrame(stats)
    
    # Save CSV
    stats_csv_path = os.path.join(RESULTS_DIR, "summary_statistics.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    
    # Save LaTeX
    if export_latex:
        stats_tex_path = os.path.join(RESULTS_DIR, "summary_statistics.txt")
        with open(stats_tex_path, 'w') as f:
            f.write(stats_df.to_latex(index=False, float_format="%.4f"))
        
    # 2. findings.txt
    findings_path = os.path.join(RESULTS_DIR, "findings.txt")
    with open(findings_path, 'w') as f:
        f.write("=== Key Findings: Reward Function Study ===\n\n")
        
        for size in df['grid_size_str'].unique():
            f.write(f"--- Grid Size: {size} ---\n")
            size_df = df[df['grid_size_str'] == size]
            
            # Compare Shaped vs Sparse
            shaped = size_df[size_df['condition'] == 'ShapedReward']
            sparse = size_df[size_df['condition'] == 'SparseReward']
            
            if not shaped.empty and not sparse.empty:
                shaped_iters = shaped[shaped['status'] == 'success']['iterations_to_converge'].mean()
                sparse_iters = sparse[sparse['status'] == 'success']['iterations_to_converge'].mean()
                
                if not np.isnan(shaped_iters) and not np.isnan(sparse_iters):
                    if shaped_iters < sparse_iters:
                        diff = (sparse_iters - shaped_iters) / sparse_iters * 100
                        f.write(f"* ShapedReward converged {diff:.1f}% faster than SparseReward on average.\n")
                    elif sparse_iters < shaped_iters:
                        diff = (shaped_iters - sparse_iters) / shaped_iters * 100
                        f.write(f"* SparseReward converged {diff:.1f}% faster than ShapedReward on average.\n")
                        
            # Analyze Deceptive failures
            decep = size_df[size_df['condition'] == 'DeceptiveReward']
            if not decep.empty:
                decep_fail = len(decep[decep['status'] != 'success'])
                failure_rate = (decep_fail / len(decep)) * 100
                f.write(f"* DeceptiveReward failed to find optimal policy (timeout) in {failure_rate:.1f}% of runs.\n")
                
                # If they succeeded, how optimal were they?
                decep_succ = decep[decep['status'] == 'success']
                if not decep_succ.empty:
                    opt = decep_succ['optimality_score'].mean()
                    f.write(f"* DeceptiveReward produced paths with an optimality score of {opt:.3f} (Lower = further away from optimal).\n")
                    
            f.write("\n")
            
        f.write("--- Statistical Significance Test ---\n")
        
        # 3. Wilcoxon signed-rank
        # Match seeds across Shaped vs Sparse over all grids
        shaped_full = df[(df['condition'] == 'ShapedReward') & (df['status'] == 'success')]
        sparse_full = df[(df['condition'] == 'SparseReward') & (df['status'] == 'success')]
        
        # We need paired samples
        paired_data = pd.merge(
            shaped_full[['grid_size_str', 'seed', 'iterations_to_converge']],
            sparse_full[['grid_size_str', 'seed', 'iterations_to_converge']],
            on=['grid_size_str', 'seed'],
            suffixes=('_shaped', '_sparse')
        )
        
        if len(paired_data) > 0:
            diffs = paired_data['iterations_to_converge_shaped'] - paired_data['iterations_to_converge_sparse']
            
            # Wilcoxon zero-diffs break scipy slightly if all are 0, handle cleanly:
            if all(diffs == 0):
                f.write("Wilcoxon Signed-Rank Test (Shaped vs Sparse Convergence): p-value = 1.0 (Identical distributions)\n")
            else:
                res = wilcoxon(paired_data['iterations_to_converge_shaped'], paired_data['iterations_to_converge_sparse'])
                f.write(f"Wilcoxon Signed-Rank Test (Shaped vs Sparse Convergence): p-value = {res.pvalue:.4g}\n")
                if res.pvalue < 0.05:
                    f.write("  -> Result is STATISTICALLY SIGNIFICANT (p < 0.05).\n")
                else:
                    f.write("  -> Result is NOT statistically significant.\n")
        else:
            f.write("Not enough paired successful runs to conduct statistical testing.\n")
            
    print(f"Findings exported to {findings_path}")

if __name__ == "__main__":
    run_analysis()
