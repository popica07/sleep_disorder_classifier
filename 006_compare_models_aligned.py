import os
import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import numpy as np
import scipy.stats

from config import PATH_TO_PROCESSED_DATA

ALIGNED_RESULTS_DIR = "/content/baseline_codebase_sleep_night_features/night_classification_results/baseline"

def load_result(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def calculate_ci(values, confidence=0.95):
    if not values:
        return 0.0, 0.0, 0.0
    
    n = len(values)
    mean = np.mean(values)
    se = scipy.stats.sem(values)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    
    return mean, mean - h, mean + h

def extract_metrics_with_ci(result_data, model_name, category, scenario_name=None):
    if result_data is None:
        return None
    
    f1_vals, auroc_vals, auprc_vals = [], [], []
    
    if 'seed_metrics' in result_data:
        for seed_key, metrics in result_data['seed_metrics'].items():
            f1_vals.append(metrics['f1'])
            auroc_vals.append(metrics['auroc'])
            auprc_vals.append(metrics['auprc'])
            
    elif 'fold_results' in result_data:
        folds = result_data['fold_results']
        f1_vals = [r['f1_macro'] for r in folds]
        if 'auroc' in folds[0]:
            auroc_vals = [r['auroc'] for r in folds]
        if 'auprc' in folds[0]:
            auprc_vals = [r['auprc'] for r in folds]

    f1_m, f1_lb, f1_ub = calculate_ci(f1_vals)
    auroc_m, auroc_lb, auroc_ub = calculate_ci(auroc_vals)
    auprc_m, auprc_lb, auprc_ub = calculate_ci(auprc_vals)
    
    return {
        "Category": category,
        "Model": model_name,
        "Scenario": scenario_name,
        "Macro F1": f1_m,
        "Macro F1 (95% CI)": f"{f1_m:.4f} ({f1_lb:.4f}-{f1_ub:.4f})",
        "AUROC (95% CI)": f"{auroc_m:.4f} ({auroc_lb:.4f}-{auroc_ub:.4f})",
        "AUPRC (95% CI)": f"{auprc_m:.4f} ({auprc_lb:.4f}-{auprc_ub:.4f})"
    }

def get_specific_metric(directory, model_base_name, category, scenario_name, is_knn=False):
    if is_knn:
        filename = f"results_knn_{scenario_name}_multiseed.pkl"
        display_name = "KNN"
    else:
        filename = f"results_{model_base_name}_{scenario_name}_multiseed.pkl"
        display_name = model_base_name.upper()
        if category == "SleepFM Embeddings" and model_base_name == "logistic":
            display_name = "Logistic Regression"

    path = os.path.join(directory, filename)
    result = load_result(path)
    return extract_metrics_with_ci(result, display_name, category, scenario_name)

def main(args):
    if args.dataset_dir is None:
        args.dataset_dir = PATH_TO_PROCESSED_DATA
        
    sleepfm_results_dir = os.path.join(args.dataset_dir, "night_classification_results", "baseline")
    
    OVERSAMPLE_LEVELS = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    
    SCENARIOS_GENERIC_BASE = ["A_None", "B_FocalWeight", "C_WeightsOnly"]
    SCENARIOS_GENERIC_OVERSAMPLE = []
    for lvl in OVERSAMPLE_LEVELS:
        pct = int(lvl * 100)
        SCENARIOS_GENERIC_OVERSAMPLE.append(f"D_F_{pct}_oversample")
        SCENARIOS_GENERIC_OVERSAMPLE.append(f"D_A_{pct}_oversample")
    
    SCENARIOS_KNN_BASE = ["A_UniformWeights", "C_DistanceWeights"]
    SCENARIOS_KNN_OVERSAMPLE = []
    for lvl in OVERSAMPLE_LEVELS:
        pct = int(lvl * 100)
        SCENARIOS_KNN_OVERSAMPLE.append(f"B_Uniform_F_{pct}_oversample")
        SCENARIOS_KNN_OVERSAMPLE.append(f"B_Uniform_A_{pct}_oversample")
        SCENARIOS_KNN_OVERSAMPLE.append(f"D_Distance_F_{pct}_oversample")
        SCENARIOS_KNN_OVERSAMPLE.append(f"D_Distance_A_{pct}_oversample")

    final_metrics = []

    final_metrics.append(get_specific_metric(ALIGNED_RESULTS_DIR, "rf", "Statistical Features", "C_WeightsOnly"))
    final_metrics.append(get_specific_metric(ALIGNED_RESULTS_DIR, "mlp", "Statistical Features", "C_WeightsOnly"))
    final_metrics.append(get_specific_metric(ALIGNED_RESULTS_DIR, "knn", "Statistical Features", "C_DistanceWeights", is_knn=True))

    final_metrics.append(get_specific_metric(sleepfm_results_dir, "logistic", "SleepFM Embeddings", "C_WeightsOnly"))
    final_metrics.append(get_specific_metric(sleepfm_results_dir, "rf", "SleepFM Embeddings", "C_WeightsOnly"))
    final_metrics.append(get_specific_metric(sleepfm_results_dir, "mlp", "SleepFM Embeddings", "C_WeightsOnly"))
    final_metrics.append(get_specific_metric(sleepfm_results_dir, "knn", "SleepFM Embeddings", "C_DistanceWeights", is_knn=True))

    final_metrics = [m for m in final_metrics if m is not None]

    if final_metrics:
        df = pd.DataFrame(final_metrics)
        df["Model"] = df["Model"] + " (" + df["Scenario"] + ")"
        
        df_disp = df.sort_values(by="Macro F1", ascending=False)
        final_cols = ["Category", "Model", "Macro F1 (95% CI)", "AUROC (95% CI)", "AUPRC (95% CI)"]
        
        print("\n" + "="*80)
        print("FINAL MODEL COMPARISON (50 Seeds Average)")
        print("="*80)
        print(df_disp[final_cols].to_markdown(index=False))
        print("="*80 + "\n")
        
        df_disp[final_cols].to_csv(os.path.join(sleepfm_results_dir, "final_comparison_table.csv"), index=False)


    sfm_metrics = []
    hc_metrics = []
    
    def create_placeholder_row(model_name, category):
        return {
            "Category": category,
            "Model": model_name,
            "Scenario": "D_Oversample_Variants",
            "Macro F1 (95% CI)": "REFER TO THE OVERSAMPLING IMPACT ANALYSIS TABLE",
            "AUROC (95% CI)": "REFER TO THE OVERSAMPLING IMPACT ANALYSIS TABLE",
            "AUPRC (95% CI)": "REFER TO THE OVERSAMPLING IMPACT ANALYSIS TABLE"
        }

    for model_name in ["logistic", "rf", "mlp"]:

        for sc in SCENARIOS_GENERIC_BASE:
            m = get_specific_metric(sleepfm_results_dir, model_name, "SleepFM Embeddings", sc)
            if m: sfm_metrics.append(m)

        sfm_metrics.append(create_placeholder_row(model_name.upper() if model_name != "logistic" else "Logistic Regression", "SleepFM Embeddings"))


    for sc in SCENARIOS_KNN_BASE:
        m = get_specific_metric(sleepfm_results_dir, "knn", "SleepFM Embeddings", sc, is_knn=True)
        if m: sfm_metrics.append(m)
    sfm_metrics.append(create_placeholder_row("KNN", "SleepFM Embeddings"))
            
    for model_name in ["rf", "mlp"]:
        for sc in SCENARIOS_GENERIC_BASE:
            m = get_specific_metric(ALIGNED_RESULTS_DIR, model_name, "Statistical Features", sc)
            if m: hc_metrics.append(m)
        hc_metrics.append(create_placeholder_row(model_name.upper(), "Statistical Features"))


    for sc in SCENARIOS_KNN_BASE:
        m = get_specific_metric(ALIGNED_RESULTS_DIR, "knn", "Statistical Features", sc, is_knn=True)
        if m: hc_metrics.append(m)
    hc_metrics.append(create_placeholder_row("KNN", "Statistical Features"))

    if sfm_metrics:
        df_sfm = pd.DataFrame(sfm_metrics)
        cols = ["Model", "Scenario", "Macro F1 (95% CI)", "AUROC (95% CI)", "AUPRC (95% CI)"]
        print("\n" + "="*80)
        print("IMBALANCE HANDLING IMPACT (SleepFM Embeddings Models)")
        print("="*80)
        print(df_sfm[cols].to_markdown(index=False))
        df_sfm[cols].to_csv(os.path.join(sleepfm_results_dir, "imbalance_impact_analysis_sleepfm.csv"), index=False)

    if hc_metrics:
        df_hc = pd.DataFrame(hc_metrics)
        cols = ["Model", "Scenario", "Macro F1 (95% CI)", "AUROC (95% CI)", "AUPRC (95% CI)"]
        print("\n" + "="*80)
        print("IMBALANCE HANDLING IMPACT (Statistic Features Models)")
        print("="*80)
        print(df_hc[cols].to_markdown(index=False))
        df_hc[cols].to_csv(os.path.join(sleepfm_results_dir, "imbalance_impact_analysis_handcrafted.csv"), index=False)

    oversample_metrics = []
    
    def scan_scenarios(models, directory, scenarios, category, prefix_knn=False):
        for model in models:
            for sc in scenarios:
                if prefix_knn:
                    m = get_specific_metric(directory, model, category, sc, is_knn=True)
                else:
                    m = get_specific_metric(directory, model, category, sc, is_knn=False)
                    
                if m:
                    m['Sort_Scenario'] = sc
                    oversample_metrics.append(m)

    scan_scenarios(["logistic", "rf", "mlp"], sleepfm_results_dir, SCENARIOS_GENERIC_OVERSAMPLE, "SleepFM", prefix_knn=False)

    scan_scenarios(["knn"], sleepfm_results_dir, SCENARIOS_KNN_OVERSAMPLE, "SleepFM", prefix_knn=True)

    scan_scenarios(["rf", "mlp"], ALIGNED_RESULTS_DIR, SCENARIOS_GENERIC_OVERSAMPLE, "Handcrafted", prefix_knn=False)
    scan_scenarios(["knn"], ALIGNED_RESULTS_DIR, SCENARIOS_KNN_OVERSAMPLE, "Handcrafted", prefix_knn=True)
    
    if oversample_metrics:
        df_os = pd.DataFrame(oversample_metrics)
        cols = ["Category", "Model", "Scenario", "Macro F1 (95% CI)", "AUROC (95% CI)"]
        df_os = df_os.sort_values(by=["Category", "Model", "Scenario"])
        
        print("\n" + "="*80)
        print("OVERSAMPLING IMPACT ANALYSIS")
        print("="*80)
        print(df_os[cols].to_markdown(index=False))
        df_os[cols].to_csv(os.path.join(sleepfm_results_dir, "oversampling_impact_analysis.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    args = parser.parse_args()
    main(args)