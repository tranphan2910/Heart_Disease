"""
Test Pure Optimized Training (No Intentional Baseline Weakening)
================================================================
Test với FULL OPTIMIZED training ngay từ đầu - không có phase "fast"
Xem XAI + FE có thực sự improve được không khi bắt đầu với hyperparameters tốt
"""

import pandas as pd
import numpy as np
from pipeline.model_trainer import ModelTrainer
from itertools import combinations

def test_pure_optimized_with_fe(csv_file):
    """Test training với optimized mode thuần túy + optional FE"""
    
    print("=" * 80)
    print("PURE OPTIMIZED TRAINING TEST")
    print("No intentional baseline weakening - Test real FE effectiveness")
    print("=" * 80)
    print()
    
    # Load data
    print("[1/3] Loading data...")
    data = pd.read_csv(csv_file)
    X = data.drop('target', axis=1)
    y = data['target']
    print(f"   Dataset: {csv_file}")
    print(f"   Samples: {len(data)}, Features: {len(X.columns)}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    print()
    
    # ============================================
    # BASELINE: Pure Optimized (No FE)
    # ============================================
    print("[2/3] Training BASELINE (Optimized, No FE)...")
    print("-" * 80)
    
    trainer_baseline = ModelTrainer(random_state=42, test_size=0.2)
    results_baseline = trainer_baseline.full_training_pipeline(X, y, enable_xai_improvement=False)
    
    baseline_acc = results_baseline['best_metrics']['Accuracy']
    baseline_model = results_baseline['best_model_name']
    baseline_f1 = results_baseline['best_metrics']['F1 Score']
    
    print()
    print(f" Baseline Result:")
    print(f"   Best Model: {baseline_model}")
    print(f"   Accuracy: {baseline_acc:.4f}")
    print(f"   F1 Score: {baseline_f1:.4f}")
    print()
    
    # ============================================
    # IMPROVED: Optimized + Feature Engineering
    # ============================================
    print("[3/3] Training IMPROVED (Optimized + Manual FE)...")
    print("-" * 80)
    print()
    
    # Create interaction features manually (simple approach)
    print("   Creating interaction features...")
    X_fe = X.copy()
    
    # Top features based on domain knowledge
    top_features = ['cp', 'thalach', 'oldpeak', 'ca', 'thal']
    
    # Create 5 pairwise interactions
    interactions = list(combinations(top_features, 2))[:5]
    new_features = []
    
    for feat1, feat2 in interactions:
        if feat1 in X_fe.columns and feat2 in X_fe.columns:
            new_col = f"{feat1}_x_{feat2}"
            X_fe[new_col] = X_fe[feat1] * X_fe[feat2]
            new_features.append(new_col)
    
    print(f"    Created {len(new_features)} interaction features")
    print(f"   Features: {len(X.columns)} → {len(X_fe.columns)}")
    print(f"   New features: {new_features[:3]}...")
    print()
    
    # Train with FE
    trainer_improved = ModelTrainer(random_state=42, test_size=0.2)
    results_improved = trainer_improved.full_training_pipeline(X_fe, y, enable_xai_improvement=False)
    
    improved_acc = results_improved['best_metrics']['Accuracy']
    improved_model = results_improved['best_model_name']
    improved_f1 = results_improved['best_metrics']['F1 Score']
    
    print()
    print(f" Improved Result:")
    print(f"   Best Model: {improved_model}")
    print(f"   Accuracy: {improved_acc:.4f}")
    print(f"   F1 Score: {improved_f1:.4f}")
    print()
    
    # ============================================
    # COMPARISON
    # ============================================
    improvement = improved_acc - baseline_acc
    improvement_pct = (improvement / baseline_acc) * 100
    
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print()
    
    print(" RESULTS:")
    print(f"   Baseline (Optimized):          {baseline_acc:.4f} ({baseline_model})")
    print(f"   Improved (Optimized + FE):     {improved_acc:.4f} ({improved_model})")
    print()
    
    print("IMPROVEMENT:")
    print(f"   Absolute: {improvement:+.4f}")
    print(f"   Relative: {improvement_pct:+.2f}%")
    print()
    
    # Detailed comparison
    print("=" * 80)
    print("DETAILED MODEL COMPARISON")
    print("=" * 80)
    print()
    
    baseline_df = results_baseline['tuned_results']
    improved_df = results_improved['tuned_results']
    
    comparison_data = []
    for model_name in baseline_df['Model Name'].values:
        baseline_row = baseline_df[baseline_df['Model Name'] == model_name].iloc[0]
        improved_row = improved_df[improved_df['Model Name'] == model_name].iloc[0]
        
        baseline_model_acc = baseline_row['Accuracy']
        improved_model_acc = improved_row['Accuracy']
        change = improved_model_acc - baseline_model_acc
        change_pct = (change / baseline_model_acc) * 100
        
        comparison_data.append({
            'Model': model_name,
            'Baseline': f"{baseline_model_acc:.4f}",
            'Improved': f"{improved_model_acc:.4f}",
            'Change': f"{change:+.4f}",
            'Change %': f"{change_pct:+.2f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()
    
    # ============================================
    # VERDICT
    # ============================================
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()
    
    if improvement > 0.001:  # > 0.1%
        print(f" FEATURE ENGINEERING WORKS!")
        print(f"   Improved performance by {improvement_pct:.2f}%")
        print(f"   This is REAL improvement (no baseline weakening)")
    elif improvement > -0.001:  # -0.1% to +0.1%
        print(f"➖ NO SIGNIFICANT CHANGE")
        print(f"   FE had minimal impact ({improvement_pct:.2f}%)")
        print(f"   Dataset may already be optimized")
    else:
        print(f"⚠️ PERFORMANCE DECREASED")
        print(f"   FE reduced performance by {abs(improvement_pct):.2f}%")
        print(f"   Original features may be sufficient")
    
    print()
    print("💡 KEY INSIGHT:")
    print(f"   This test used FULL OPTIMIZED training from start")
    print(f"   No intentional baseline weakening")
    print(f"   Result shows TRUE FE effectiveness on this dataset")
    print()
    print("=" * 80)
    
    return {
        'baseline_acc': baseline_acc,
        'improved_acc': improved_acc,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'comparison_df': comparison_df
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "data/heart-disease-UCI.csv"
    
    results = test_pure_optimized_with_fe(csv_file)
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline (Optimized):     {results['baseline_acc']:.4f}")
    print(f"Improved (Optimized+FE):  {results['improved_acc']:.4f}")
    print(f"Improvement:              {results['improvement']:+.4f} ({results['improvement_pct']:+.2f}%)")
    print("=" * 80)
