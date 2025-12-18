"""
Compare classification results between baseline and macro feature models.

This script compares the performance of classification models trained on:
1. Baseline features (from model_comparision_classification.ipynb)
2. Macro features (from model_comparision_classification_macro.ipynb)

It performs statistical hypothesis tests to determine if improvements are significant.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Statistical test imports
from scipy import stats
from scipy.stats import chi2
from sklearn.metrics import confusion_matrix



def load_results_from_notebooks(
    baseline_results: Dict,
    macro_results: Dict,
    baseline_y_true: np.ndarray,
    macro_y_true: np.ndarray,
) -> Dict[str, Dict]:
    """
    Organize results from both notebooks for comparison.
    
    Parameters
    ----------
    baseline_results : Dict
        Results dictionary from baseline notebook (keys: model_name -> metrics)
    macro_results : Dict
        Results dictionary from macro notebook (keys: model_name -> metrics)
    baseline_y_true : np.ndarray
        True labels from baseline test set
    macro_y_true : np.ndarray
        True labels from macro test set
    
    Returns
    -------
    Dict
        Organized results for comparison
    """
    comparison_data = {}
    
    # Get common models
    baseline_models = set(baseline_results.keys())
    macro_models = set(macro_results.keys())
    common_models = baseline_models & macro_models
    
    for model_name in common_models:
        baseline_result = baseline_results[model_name]
        macro_result = macro_results[model_name]
        
        comparison_data[model_name] = {
            'baseline': {
                'accuracy': baseline_result.get('accuracy', np.nan),
                'precision': baseline_result.get('precision', np.nan),
                'recall': baseline_result.get('recall', np.nan),
                'f1': baseline_result.get('f1', np.nan),
                'auc': baseline_result.get('auc', np.nan),
                'y_pred': baseline_result.get('y_pred', None),
                'y_proba': baseline_result.get('y_proba', None),
            },
            'macro': {
                'accuracy': macro_result.get('accuracy', np.nan),
                'precision': macro_result.get('precision', np.nan),
                'recall': macro_result.get('recall', np.nan),
                'f1': macro_result.get('f1', np.nan),
                'auc': macro_result.get('auc', np.nan),
                'y_pred': macro_result.get('y_pred', None),
                'y_proba': macro_result.get('y_proba', None),
            },
            'baseline_y_true': baseline_y_true,
            'macro_y_true': macro_y_true,
        }
    
    return comparison_data


def create_comparison_table(comparison_data: Dict) -> pd.DataFrame:
    """
    Create a comparison table showing metrics from both approaches.
    
    Parameters
    ----------
    comparison_data : Dict
        Organized comparison data from load_results_from_notebooks
    
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics and differences
    """
    rows = []
    
    for model_name, data in comparison_data.items():
        baseline = data['baseline']
        macro = data['macro']
        
        row = {
            'Model': model_name.replace('_', ' ').title(),
            'Baseline_Accuracy': baseline['accuracy'],
            'Macro_Accuracy': macro['accuracy'],
            'Accuracy_Diff': macro['accuracy'] - baseline['accuracy'],
            'Baseline_Precision': baseline['precision'],
            'Macro_Precision': macro['precision'],
            'Precision_Diff': macro['precision'] - baseline['precision'],
            'Baseline_Recall': baseline['recall'],
            'Macro_Recall': macro['recall'],
            'Recall_Diff': macro['recall'] - baseline['recall'],
            'Baseline_F1': baseline['f1'],
            'Macro_F1': macro['f1'],
            'F1_Diff': macro['f1'] - baseline['f1'],
            'Baseline_AUC': baseline['auc'],
            'Macro_AUC': macro['auc'],
            'AUC_Diff': macro['auc'] - baseline['auc'],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def mcnemar_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    correction: bool = True
) -> Tuple[float, float]:
    """
    Perform McNemar's test to compare two classification models.
    
    McNemar's test is a paired non-parametric test used to determine if there are 
    differences on a dichotomous dependent variable between two related groups.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred1 : np.ndarray
        Predictions from first model (baseline)
    y_pred2 : np.ndarray
        Predictions from second model (macro)
    correction : bool, default=True
        Use continuity correction (Yates' correction)
    
    Returns
    -------
    Tuple[float, float]
        (statistic, p-value)
    """
    # Ensure same length
    min_len = min(len(y_true), len(y_pred1), len(y_pred2))
    y_true = y_true[:min_len]
    y_pred1 = y_pred1[:min_len]
    y_pred2 = y_pred2[:min_len]
    
    # Create contingency table
    # Model 1 correct, Model 2 correct: a
    # Model 1 correct, Model 2 wrong: b
    # Model 1 wrong, Model 2 correct: c
    # Model 1 wrong, Model 2 wrong: d
    
    model1_correct = (y_pred1 == y_true)
    model2_correct = (y_pred2 == y_true)
    
    b = np.sum(model1_correct & ~model2_correct)  # Model 1 correct, Model 2 wrong
    c = np.sum(~model1_correct & model2_correct)  # Model 1 wrong, Model 2 correct
    
    # McNemar's test statistic
    # With continuity correction (Yates' correction):
    # chi^2 = (|b - c| - 1)^2 / (b + c)
    # Without correction:
    # chi^2 = (b - c)^2 / (b + c)
    
    if b + c == 0:
        # No discordant pairs - models agree perfectly
        return 0.0, 1.0
    
    if correction:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        statistic = (b - c) ** 2 / (b + c)
    
    # P-value from chi-squared distribution with 1 degree of freedom
    pvalue = 1 - chi2.cdf(statistic, df=1)
    
    return statistic, pvalue


def delong_test(y_true: np.ndarray, y_proba1: np.ndarray, y_proba2: np.ndarray) -> Tuple[float, float]:
    """
    Perform DeLong's test to compare AUC-ROC between two models.
    
    This is a simplified implementation. For production, consider using
    more robust implementations like delong_roc_test from pyroc.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba1 : np.ndarray
        Predicted probabilities from first model
    y_proba2 : np.ndarray
        Predicted probabilities from second model
    
    Returns
    -------
    Tuple[float, float]
        (z-score, p-value)
    """
    from sklearn.metrics import roc_auc_score
    
    # Ensure same length and extract probabilities
    min_len = min(len(y_true), len(y_proba1), len(y_proba2))
    y_true = y_true[:min_len]
    
    # Extract positive class probabilities
    if len(y_proba1.shape) > 1:
        y_proba1 = y_proba1[:min_len, 1] if y_proba1.shape[1] > 1 else y_proba1[:min_len].flatten()
    else:
        y_proba1 = y_proba1[:min_len].flatten()
    
    if len(y_proba2.shape) > 1:
        y_proba2 = y_proba2[:min_len, 1] if y_proba2.shape[1] > 1 else y_proba2[:min_len].flatten()
    else:
        y_proba2 = y_proba2[:min_len].flatten()
    
    # Calculate AUCs
    auc1 = roc_auc_score(y_true, y_proba1)
    auc2 = roc_auc_score(y_true, y_proba2)
    
    # Simplified DeLong test using bootstrap approximation
    # For exact test, consider using pyroc library
    n_bootstrap = 1000
    auc1_bootstrap = []
    auc2_bootstrap = []
    
    np.random.seed(42)
    n_samples = len(y_true)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_proba1_boot = y_proba1[indices]
        y_proba2_boot = y_proba2[indices]
        
        try:
            auc1_bootstrap.append(roc_auc_score(y_true_boot, y_proba1_boot))
            auc2_bootstrap.append(roc_auc_score(y_true_boot, y_proba2_boot))
        except:
            continue
    
    if len(auc1_bootstrap) < 10:
        # Fallback to simple difference test
        diff = auc2 - auc1
        se = np.sqrt(0.25 / n_samples)  # Conservative SE estimate
        z_score = diff / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        return z_score, p_value
    
    # Bootstrap-based test
    diff = np.array(auc2_bootstrap) - np.array(auc1_bootstrap)
    se = np.std(diff)
    z_score = (auc2 - auc1) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return z_score, p_value


def perform_hypothesis_tests(comparison_data: Dict) -> pd.DataFrame:
    """
    Perform statistical hypothesis tests for each model comparison.
    
    Parameters
    ----------
    comparison_data : Dict
        Organized comparison data
    
    Returns
    -------
    pd.DataFrame
        Results of hypothesis tests
    """
    test_results = []
    
    for model_name, data in comparison_data.items():
        baseline = data['baseline']
        macro = data['macro']
        baseline_y_true = data['baseline_y_true']
        macro_y_true = data['macro_y_true']
        
        baseline_y_pred = baseline.get('y_pred')
        macro_y_pred = macro.get('y_pred')
        baseline_y_proba = baseline.get('y_proba')
        macro_y_proba = macro.get('y_proba')
        
        # Check if we can perform paired tests (same test set)
        # If test sets have same length, we can do McNemar's test
        can_do_mcnemar = False
        can_do_delong = False
        
        if (baseline_y_true is not None and macro_y_true is not None and 
            baseline_y_pred is not None and macro_y_pred is not None):
            # Check if test sets are compatible
            min_true_len = min(len(baseline_y_true), len(macro_y_true))
            min_pred_len = min(len(baseline_y_pred), len(macro_y_pred))
            
            # If lengths match or are very close, we can align them
            if abs(len(baseline_y_true) - len(macro_y_true)) <= 10 and min_pred_len >= 100:
                can_do_mcnemar = True
        
        if (baseline_y_true is not None and macro_y_true is not None and 
            baseline_y_proba is not None and macro_y_proba is not None):
            min_true_len = min(len(baseline_y_true), len(macro_y_true))
            # Extract probability lengths
            if len(baseline_y_proba.shape) > 1:
                baseline_proba_len = baseline_y_proba.shape[0]
            else:
                baseline_proba_len = len(baseline_y_proba)
            if len(macro_y_proba.shape) > 1:
                macro_proba_len = macro_y_proba.shape[0]
            else:
                macro_proba_len = len(macro_y_proba)
            
            min_proba_len = min(baseline_proba_len, macro_proba_len)
            if abs(baseline_proba_len - macro_proba_len) <= 10 and min_proba_len >= 100:
                can_do_delong = True
        
        # Perform McNemar's test if possible
        if can_do_mcnemar:
            try:
                # Align to minimum length
                min_len = min(len(baseline_y_true), len(macro_y_true), 
                             len(baseline_y_pred), len(macro_y_pred))
                y_true_aligned = baseline_y_true[:min_len] if len(baseline_y_true) >= min_len else macro_y_true[:min_len]
                baseline_y_pred_aligned = baseline_y_pred[:min_len]
                macro_y_pred_aligned = macro_y_pred[:min_len]
                
                mcnemar_stat, mcnemar_p = mcnemar_test(
                    y_true_aligned,
                    baseline_y_pred_aligned,
                    macro_y_pred_aligned
                )
            except Exception as e:
                mcnemar_stat, mcnemar_p = np.nan, np.nan
        else:
            mcnemar_stat, mcnemar_p = np.nan, np.nan
        
        # Perform DeLong's test if possible
        if can_do_delong:
            try:
                # Align to minimum length
                if len(baseline_y_proba.shape) > 1:
                    baseline_proba_len = baseline_y_proba.shape[0]
                else:
                    baseline_proba_len = len(baseline_y_proba)
                if len(macro_y_proba.shape) > 1:
                    macro_proba_len = macro_y_proba.shape[0]
                else:
                    macro_proba_len = len(macro_y_proba)
                
                min_len = min(len(baseline_y_true), len(macro_y_true), 
                             baseline_proba_len, macro_proba_len)
                
                # Use baseline test set if available, otherwise macro
                y_true_for_auc = baseline_y_true[:min_len] if baseline_y_true is not None else macro_y_true[:min_len]
                
                # Align probabilities
                if len(baseline_y_proba.shape) > 1:
                    baseline_proba_aligned = baseline_y_proba[:min_len, :]
                else:
                    baseline_proba_aligned = baseline_y_proba[:min_len]
                
                if len(macro_y_proba.shape) > 1:
                    macro_proba_aligned = macro_y_proba[:min_len, :]
                else:
                    macro_proba_aligned = macro_y_proba[:min_len]
                
                delong_z, delong_p = delong_test(
                    y_true_for_auc,
                    baseline_proba_aligned,
                    macro_proba_aligned
                )
            except Exception as e:
                delong_z, delong_p = np.nan, np.nan
        else:
            delong_z, delong_p = np.nan, np.nan
        
        # Calculate improvements (these can always be calculated from metrics)
        accuracy_improvement = macro['accuracy'] - baseline['accuracy']
        f1_improvement = macro['f1'] - baseline['f1']
        auc_improvement = macro['auc'] - baseline['auc']
        
        test_results.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy_Improvement': accuracy_improvement,
            'F1_Improvement': f1_improvement,
            'AUC_Improvement': auc_improvement,
            'McNemar_Statistic': mcnemar_stat,
            'McNemar_PValue': mcnemar_p,
            'McNemar_Significant': mcnemar_p < 0.05 if not np.isnan(mcnemar_p) else False,
            'DeLong_ZScore': delong_z,
            'DeLong_PValue': delong_p,
            'DeLong_Significant': delong_p < 0.05 if not np.isnan(delong_p) else False,
        })
    
    return pd.DataFrame(test_results)


def visualize_comparison(comparison_df: pd.DataFrame, test_results_df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Create visualizations comparing baseline and macro models.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison table with metrics
    test_results_df : pd.DataFrame
        Hypothesis test results
    save_path : Optional[Path]
        Path to save figures
    """
    # Set style
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Accuracy Comparison
    ax1 = plt.subplot(2, 3, 1)
    x_pos = np.arange(len(comparison_df))
    width = 0.35
    ax1.bar(x_pos - width/2, comparison_df['Baseline_Accuracy'], width, label='Baseline', alpha=0.8)
    ax1.bar(x_pos + width/2, comparison_df['Macro_Accuracy'], width, label='Macro Features', alpha=0.8)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison: Baseline vs Macro Features')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. F1-Score Comparison
    ax2 = plt.subplot(2, 3, 2)
    ax2.bar(x_pos - width/2, comparison_df['Baseline_F1'], width, label='Baseline', alpha=0.8)
    ax2.bar(x_pos + width/2, comparison_df['Macro_F1'], width, label='Macro Features', alpha=0.8)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score Comparison: Baseline vs Macro Features')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. AUC-ROC Comparison
    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(x_pos - width/2, comparison_df['Baseline_AUC'], width, label='Baseline', alpha=0.8)
    ax3.bar(x_pos + width/2, comparison_df['Macro_AUC'], width, label='Macro Features', alpha=0.8)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('AUC-ROC')
    ax3.set_title('AUC-ROC Comparison: Baseline vs Macro Features')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Improvement in Metrics
    ax4 = plt.subplot(2, 3, 4)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    improvements = [
        comparison_df['Accuracy_Diff'].mean(),
        comparison_df['Precision_Diff'].mean(),
        comparison_df['Recall_Diff'].mean(),
        comparison_df['F1_Diff'].mean(),
        comparison_df['AUC_Diff'].mean(),
    ]
    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax4.barh(metrics, improvements, color=colors, alpha=0.7)
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Average Improvement (Macro - Baseline)')
    ax4.set_title('Average Improvement Across All Models')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Statistical Significance (P-values)
    ax5 = plt.subplot(2, 3, 5)
    models = test_results_df['Model']
    mcnemar_p = test_results_df['McNemar_PValue'].replace([np.inf, -np.inf], np.nan)
    delong_p = test_results_df['DeLong_PValue'].replace([np.inf, -np.inf], np.nan)
    
    # Filter out NaN values for plotting
    valid_mask = mcnemar_p.notna() | delong_p.notna()
    if valid_mask.sum() > 0:
        x_pos_p = np.arange(len(models[valid_mask]))
        models_filtered = models[valid_mask]
        mcnemar_p_filtered = mcnemar_p[valid_mask]
        delong_p_filtered = delong_p[valid_mask]
        
        # Plot only valid p-values
        mcnemar_valid = mcnemar_p_filtered.notna()
        delong_valid = delong_p_filtered.notna()
        
        if mcnemar_valid.sum() > 0:
            ax5.bar(x_pos_p[mcnemar_valid] - width/2, mcnemar_p_filtered[mcnemar_valid], 
                   width, label="McNemar's Test", alpha=0.8)
        if delong_valid.sum() > 0:
            ax5.bar(x_pos_p[delong_valid] + width/2, delong_p_filtered[delong_valid], 
                   width, label="DeLong's Test", alpha=0.8)
        
        ax5.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        ax5.set_xlabel('Model')
        ax5.set_ylabel('P-Value')
        ax5.set_title('Statistical Significance Tests')
        ax5.set_xticks(x_pos_p)
        ax5.set_xticklabels(models_filtered, rotation=45, ha='right')
        ax5.set_yscale('log')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'Tests not applicable\n(test sets incompatible)', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Statistical Significance Tests')
    
    # 6. Improvement Summary
    ax6 = plt.subplot(2, 3, 6)
    improvement_summary = test_results_df[['Model', 'Accuracy_Improvement', 'F1_Improvement', 'AUC_Improvement']]
    improvement_summary = improvement_summary.set_index('Model')
    improvement_summary.plot(kind='barh', ax=ax6, alpha=0.7)
    ax6.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax6.set_xlabel('Improvement (Macro - Baseline)')
    ax6.set_title('Metric Improvements by Model')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'classification_comparison_results.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def print_comparison_summary(comparison_df: pd.DataFrame, test_results_df: pd.DataFrame):
    """
    Print a comprehensive summary of the comparison.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison table
    test_results_df : pd.DataFrame
        Hypothesis test results
    """
    print("=" * 100)
    print("CLASSIFICATION MODEL COMPARISON: BASELINE vs MACRO FEATURES")
    print("=" * 100)
    print()
    
    # Note about test sets
    print("NOTE:")
    print("-" * 100)
    print("Baseline models use: lopbdy_features.csv (price features only)")
    print("Macro models use: all_features_with_macro.csv (price + macro features)")
    print("\n⚠ IMPORTANT: If test sets differ, paired prediction tests (McNemar's) may not be applicable.")
    print("  Metric comparisons (Accuracy, F1, AUC differences) are still valid.")
    print("  Statistical tests require compatible test set sizes.")
    print()
    
    # Summary statistics
    print("SUMMARY STATISTICS:")
    print("-" * 100)
    print(f"Average Accuracy Improvement: {comparison_df['Accuracy_Diff'].mean():.4f}")
    print(f"Average F1-Score Improvement: {comparison_df['F1_Diff'].mean():.4f}")
    print(f"Average AUC-ROC Improvement: {comparison_df['AUC_Diff'].mean():.4f}")
    print()
    
    # Best improvements
    print("BEST IMPROVEMENTS:")
    print("-" * 100)
    best_accuracy = comparison_df.loc[comparison_df['Accuracy_Diff'].idxmax()]
    best_f1 = comparison_df.loc[comparison_df['F1_Diff'].idxmax()]
    best_auc = comparison_df.loc[comparison_df['AUC_Diff'].idxmax()]
    
    print(f"Best Accuracy Improvement: {best_accuracy['Model']} ({best_accuracy['Accuracy_Diff']:.4f})")
    print(f"Best F1-Score Improvement: {best_f1['Model']} ({best_f1['F1_Diff']:.4f})")
    print(f"Best AUC-ROC Improvement: {best_auc['Model']} ({best_auc['AUC_Diff']:.4f})")
    print()
    
    # Statistical significance
    print("STATISTICAL SIGNIFICANCE TESTS:")
    print("-" * 100)
    
    # Check how many tests were actually performed
    mcnemar_performed = test_results_df['McNemar_PValue'].notna().sum()
    delong_performed = test_results_df['DeLong_PValue'].notna().sum()
    
    if mcnemar_performed > 0:
        significant_mcnemar = test_results_df[test_results_df['McNemar_Significant']]
        print(f"McNemar's Test Results (Tests performed: {mcnemar_performed}/{len(test_results_df)}):")
        print(f"  Models with statistically significant differences (p < 0.05): {len(significant_mcnemar)}")
        if len(significant_mcnemar) > 0:
            for _, row in significant_mcnemar.iterrows():
                improvement = comparison_df[comparison_df['Model'] == row['Model']]['Accuracy_Diff'].iloc[0]
                direction = "Macro better" if improvement > 0 else "Baseline better"
                print(f"    - {row['Model']}: p = {row['McNemar_PValue']:.4f} ({direction})")
        else:
            print("    No models show statistically significant differences in predictions.")
        print()
    else:
        print("McNemar's Test: Not performed (test sets may be incompatible)")
        print()
    
    if delong_performed > 0:
        significant_delong = test_results_df[test_results_df['DeLong_Significant']]
        print(f"DeLong's Test Results (Tests performed: {delong_performed}/{len(test_results_df)}):")
        print(f"  Models with statistically significant AUC differences (p < 0.05): {len(significant_delong)}")
        if len(significant_delong) > 0:
            for _, row in significant_delong.iterrows():
                improvement = comparison_df[comparison_df['Model'] == row['Model']]['AUC_Diff'].iloc[0]
                direction = "Macro better" if improvement > 0 else "Baseline better"
                print(f"    - {row['Model']}: p = {row['DeLong_PValue']:.4f} ({direction})")
        else:
            print("    No models show statistically significant differences in AUC.")
        print()
    else:
        print("DeLong's Test: Not performed (test sets may be incompatible)")
        print()
    
    # Detailed comparison table
    print("DETAILED COMPARISON TABLE:")
    print("-" * 100)
    display_df = comparison_df.copy()
    for col in display_df.columns:
        if col != 'Model' and 'Diff' not in col:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
        elif 'Diff' in col:
            display_df[col] = display_df[col].apply(lambda x: f"{x:+.4f}" if not np.isnan(x) else "N/A")
    print(display_df.to_string(index=False))
    print()
    
    # Hypothesis test results
    print("HYPOTHESIS TEST RESULTS:")
    print("-" * 100)
    test_display = test_results_df.copy()
    for col in test_display.columns:
        if col not in ['Model', 'McNemar_Significant', 'DeLong_Significant']:
            if 'PValue' in col or 'Improvement' in col:
                test_display[col] = test_display[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
            else:
                test_display[col] = test_display[col].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A")
    print(test_display.to_string(index=False))
    print()
    
    print("=" * 100)


if __name__ == "__main__":
    print("Classification Results Comparison Script")
    print("=" * 80)
    print("\nThis script compares baseline and macro feature classification models.")
    print("Run this from a notebook after executing both classification notebooks.")
    print("\nExample usage in notebook:")
    print("  from src.evaluation.compare_classification_results import *")
    print("  comparison_data = load_results_from_notebooks(baseline_results, macro_results, baseline_y_true, macro_y_true)")
    print("  comparison_df = create_comparison_table(comparison_data)")
    print("  test_results_df = perform_hypothesis_tests(comparison_data)")
    print("  print_comparison_summary(comparison_df, test_results_df)")
    print("  visualize_comparison(comparison_df, test_results_df)")

