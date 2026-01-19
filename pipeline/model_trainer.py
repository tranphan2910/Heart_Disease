import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report, 
                            accuracy_score, precision_score, recall_score, f1_score)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from .xai_explainer import XAIExplainer
from .model_improver import ModelImprover, InteractionFeatureEngine
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Training và tuning các models với pipeline tự động
    """
    
    @staticmethod
    def explain_improvement_result(baseline_acc, improved_acc, n_features_removed, features_removed):
        """
        Automatically explain improvement results
        
        Returns:
            explanation: Detailed explanation string
        """
        improvement = improved_acc - baseline_acc
        improvement_pct = (improvement / baseline_acc) * 100
        
        explanation = []
        
        if improvement > 0.01:  # Increase > 1%
            explanation.append(" **Result: SIGNIFICANT IMPROVEMENT**")
            explanation.append(f"   - Accuracy increased: {improvement:+.4f} ({improvement_pct:+.2f}%)")
            explanation.append(f"   - Successfully removed {n_features_removed} noisy features")
            explanation.append("")
            explanation.append(" **Reasons:**")
            explanation.append("   - Removed features had low individual importance")
            explanation.append("   - Removal helps model focus on important signals")
            explanation.append("   - Reduces overfitting, improves generalization")
            
        elif improvement > -0.01:  # Change within -1% to +1%
            explanation.append("➖ **Result: NO SIGNIFICANT CHANGE**")
            explanation.append(f"   - Accuracy change: {improvement:+.4f} ({improvement_pct:+.2f}%)")
            explanation.append(f"   - Removed {n_features_removed} features")
            explanation.append("")
            explanation.append(" **Reasons:**")
            explanation.append("   - Removed features had minimal impact on predictions")
            explanation.append("   - Model can perform similarly with fewer features")
            explanation.append("   - Benefits: Reduced complexity, faster inference")
            
        else:  # Decrease > 1%
            explanation.append("⚠️ **Result: PERFORMANCE DECREASED**")
            explanation.append(f"   - Accuracy decreased: {improvement:+.4f} ({improvement_pct:+.2f}%)")
            explanation.append(f"   - Removed {n_features_removed} features")
            explanation.append("")
            explanation.append(" **Possible Reasons:**")
            explanation.append("   1. **Ensemble Effect**: Features may have low individual importance")
            explanation.append("      but are crucial for ensemble models (RandomForest, XGBoost)")
            explanation.append("")
            explanation.append("   2. **Feature Interactions**: Removed features interact with others")
            explanation.append("      to create important patterns")
            explanation.append("")
            explanation.append("   3. **Noise Reduction**: Features help prevent overfitting")
            explanation.append("      by providing natural regularization")
            explanation.append("")
            explanation.append("   4. **Dataset Size**: Small datasets (< 15 features) are often")
            explanation.append("      well feature-engineered; removal causes information loss")
            explanation.append("")
            explanation.append("💡 **Recommendation**: KEEP BASELINE MODEL (no improvement)")
        
        if features_removed:
            explanation.append("")
            explanation.append(f"🗑️ **Removed Features:** {', '.join(features_removed[:5])}")
            if len(features_removed) > 5:
                explanation.append(f"   (and {len(features_removed) - 5} more...)")
        
        return "\n".join(explanation)
    
    def __init__(self, random_state=42, test_size=0.2, cv_folds=5):
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.models = {}
        self.results = []
        self.best_models = {}
        
        # Define models và param grids - Full optimized for production
        self.model_configs = {
                "MLP": (
                    MLPClassifier(random_state=random_state),
                    {
                        "hidden_layer_sizes": [(100,), (100, 50)],
                        "activation": ['relu', 'tanh'],
                        "alpha": [0.0001, 0.001],
                        "learning_rate": ['constant', 'adaptive'],
                        "solver": ['adam'],
                        "max_iter": [500]
                    }
                ),
                "GaussianNB": (
                    GaussianNB(),
                    {"var_smoothing": [1e-9, 1e-8, 1e-7]}
                ),
                "LogisticRegression": (
                    LogisticRegression(),
                    {
                        "C": [0.01, 0.1, 1, 10],
                        "solver": ['lbfgs'],
                        "penalty": ['l2'],
                        "max_iter": [500]
                    }
                ),
                "XGBoost": (
                    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                    {
                        "n_estimators": [100, 200],
                        "max_depth": [3, 5],
                        "learning_rate": [0.01, 0.1],
                        "subsample": [0.8, 1.0]
                    }
                ),
                "RandomForest": (
                    RandomForestClassifier(random_state=random_state),
                    {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10],
                        "min_samples_split": [2, 5],
                        "min_samples_leaf": [1, 2]
                    }
                ),
                "GradientBoosting": (
                    GradientBoostingClassifier(random_state=random_state),
                    {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5],
                        "subsample": [0.8, 1.0]
                    }
                )
            }
    
    def split_and_scale(self, X, y):
        """Split data và scale features"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
    def evaluate_model(self, classifier, model_name, X_train, y_train, X_test, y_test, already_fitted=False):
        """Đánh giá model và trả về metrics"""
        # Train only if model hasn't been fitted yet (e.g., baseline models)
        if not already_fitted:
            classifier.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='macro')
        recall = recall_score(y_test, y_test_pred, average='macro')
        f1 = f1_score(y_test, y_test_pred, average='macro')
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        return {
            "Model Name": model_name,
            "Training Accuracy": train_accuracy,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion Matrix": conf_matrix,
            "Classification Report": class_report
        }
    
    def train_baseline_models(self, X_train_scaled, y_train, X_test_scaled, y_test):
        """Train baseline models without tuning"""
        baseline_results = []
        
        for name, (model, _) in self.model_configs.items():
            print(f"Training baseline {name}...")
            result = self.evaluate_model(
                model, name, X_train_scaled, y_train, X_test_scaled, y_test
            )
            baseline_results.append(result)
        
        return pd.DataFrame(baseline_results)
    
    def train_and_tune_models(self, X_train_scaled, y_train, X_test_scaled, y_test):
        """Train models với hyperparameter tuning"""
        self.results = []
        
        for name, (model, param_grid) in self.model_configs.items():
            print(f"Tuning {name}...")
            
            grid = GridSearchCV(
                model, param_grid, 
                cv=self.cv_folds, 
                scoring='f1_macro', 
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train_scaled, y_train)
            best_model = grid.best_estimator_
            
            print(f"  Best params: {grid.best_params_}")
            
            # Store best model
            self.best_models[name] = best_model
            
            # Evaluate (model already fitted by GridSearchCV)
            result = self.evaluate_model(
                best_model, name, X_train_scaled, y_train, X_test_scaled, y_test, already_fitted=True
            )
            result['Best Params'] = grid.best_params_
            self.results.append(result)
        
        return pd.DataFrame(self.results)
    
    def get_best_model(self, metric='Accuracy'):
        """Lấy model tốt nhất theo metric"""
        if not self.results:
            raise ValueError("No trained models found. Run train_and_tune_models first.")
        
        results_df = pd.DataFrame(self.results)
        best_idx = results_df[metric].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model Name']
        
        return self.best_models[best_model_name], best_model_name, results_df.loc[best_idx]
    
    def full_training_pipeline(self, X, y, enable_xai_improvement=False):
        """
        Pipeline hoàn chỉnh: Split -> Scale -> Train -> Tune -> XAI Improvement -> Evaluate
        
        Args:
            enable_xai_improvement: If True, automatically improve models using XAI insights
        
        Returns:
            results_dict: Dictionary chứa tất cả kết quả
        """
        print("=" * 60)
        print("Starting Full Training Pipeline...")
        if enable_xai_improvement:
            print(" XAI-Guided Model Improvement: ENABLED")
        print("=" * 60)
        
        # 1. Split and scale
        print("\n[1/3] Splitting and scaling data...")
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = self.split_and_scale(X, y)
        print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # ====== PATIENT LOCATOR (TEST → ORIGINAL DATASET) ======
        n_show = min(7, len(X_test))

        patient_locator = pd.DataFrame({
            "Patient (Test)": [f"P{str(i).zfill(2)}" for i in range(1, n_show + 1)],
            "Dataset index": X_test.index[:n_show].to_list(),                 # index gốc (0-based)
            "Excel row": (X_test.index[:n_show].to_numpy() + 2).astype(int)   # +2 vì header + 1-based
        })

        print("\n" + "=" * 54)
        print("PATIENT LOCATOR (Test Set → Original Dataset)")
        print("=" * 54)
        print(patient_locator.to_string(index=False))
        print("=" * 54 + "\n")
        # =======================================================

        
        # 2. Baseline training (optional - quick check)
        print("\n[2/3] Training baseline models...")
        baseline_results = self.train_baseline_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # 3. Hyperparameter tuning
        print("\n[3/3] Hyperparameter tuning...")
        tuned_results = self.train_and_tune_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # 4. Get best model
        best_model, best_model_name, best_metrics = self.get_best_model('F1 Score')
        
        print("\n" + "=" * 60)
        print(f"Best Model: {best_model_name}")
        print(f"  Accuracy: {best_metrics['Accuracy']:.4f}")
        print(f"  Precision: {best_metrics['Precision']:.4f}")
        print(f"  Recall: {best_metrics['Recall']:.4f}")
        print(f"  F1 Score: {best_metrics['F1 Score']:.4f}")
        print("=" * 60)
        
        # Prepare base results
        results = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'baseline_results': baseline_results,
            'tuned_results': tuned_results,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'best_metrics': best_metrics,
            'all_models': self.best_models
        }
        
        # 5. XAI-Guided Model Improvement (NEW!)
        if enable_xai_improvement:
            try:
                print("\n" + "="*60)
                print("XAI-GUIDED MODEL IMPROVEMENT")
                print("="*60)
                
                # Run XAI analysis on baseline best model first
                print("\n[Step 1/3] Running XAI Analysis...")
                xai = XAIExplainer(
                    model=best_model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    X_train_scaled=X_train_scaled,
                    X_test_scaled=X_test_scaled,
                    feature_names=X.columns.tolist()
                )
                xai_results = xai.full_xai_pipeline()
                
                # Apply Feature Engineering based on XAI insights
                print("\n[Step 2/3] Applying Feature Engineering...")
                from itertools import combinations
                
                # Extract top features from XAI
                perm_importance = xai_results['permutation_importance']
                top_features = perm_importance.nlargest(5, 'Importance')['Feature'].tolist()
                print(f"   Top features for interactions: {top_features[:3]}...")
                
                # Create interaction features using InteractionFeatureEngine
                X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
                X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
                
                fe_engine = InteractionFeatureEngine()
                fe_engine.fit(X_train_df, xai_results)
                
                X_train_df = fe_engine.transform(X_train_df)
                X_test_df = fe_engine.transform(X_test_df)
                
                new_features = fe_engine.new_feature_names
                
                print(f"    Created {len(new_features)} interaction features")
                # Print interaction features explanation to Console as requested by user
                print("    > Top features from Origin Model used to create interactions:")
                for i, feat in enumerate(top_features, 1):
                    print(f"      {i}. {feat}")
                
                print("    > Interaction Features Details:")
                for i, feat in enumerate(new_features, 1):
                    print(f"      {i}. {feat}")
                    
                print(f"   Features: {len(X.columns)} → {len(X_train_df.columns)}")
                
                # Train improved models with FE
                print("\n[Step 3/3] Training Improved Models (Optimized + FE)...")
                X_train_improved = X_train_df.values
                X_test_improved = X_test_df.values
                
                # Train with optimized hyperparameters
                improved_results_list = []
                improved_models = {}
                
                for name, (model, param_grid) in self.model_configs.items():
                    print(f"   Training improved {name}...")
                    
                    grid = GridSearchCV(
                        model, param_grid,
                        cv=self.cv_folds,
                        scoring='f1_macro',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid.fit(X_train_improved, y_train)
                    improved_model = grid.best_estimator_
                    improved_models[name] = improved_model
                    
                    # Evaluate
                    y_pred = improved_model.predict(X_test_improved)
                    improved_acc = accuracy_score(y_test, y_pred)
                    improved_f1 = f1_score(y_test, y_pred, average='macro')
                    
                    improved_results_list.append({
                        'Model Name': name,
                        'Accuracy': improved_acc,
                        'F1 Score': improved_f1
                    })
                
                improved_results_df = pd.DataFrame(improved_results_list)
                
                # Get best improved model
                best_improved_idx = improved_results_df['F1 Score'].idxmax()
                best_improved_name = improved_results_df.loc[best_improved_idx, 'Model Name']
                best_improved_acc = improved_results_df.loc[best_improved_idx, 'Accuracy']
                best_improved_model = improved_models[best_improved_name]
                
                # Create comparison table
                print("\n" + "="*60)
                print(" COMPARISON: BASELINE vs IMPROVED")
                print("="*60)
                print()
                
                comparison_data = []
                for name in tuned_results['Model Name']:
                    baseline_acc = tuned_results[tuned_results['Model Name'] == name]['Accuracy'].values[0]
                    improved_row = improved_results_df[improved_results_df['Model Name'] == name]
                    
                    if len(improved_row) > 0:
                        improved_acc = improved_row['Accuracy'].values[0]
                        change = improved_acc - baseline_acc
                        change_pct = (change / baseline_acc) * 100
                        
                        comparison_data.append({
                            'Model': name,
                            'Baseline': f"{baseline_acc:.4f}",
                            'Improved': f"{improved_acc:.4f}",
                            'Change': f"{change:+.4f}",
                            'Change %': f"{change_pct:+.2f}%"
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                print(comparison_df.to_string(index=False))
                print()
                
                # AUTO-RANKING: Compare baseline best vs improved best
                print("="*60)
                print(" AUTO-RANKING & MODEL SELECTION")
                print("="*60)
                
                baseline_best_name = results['best_model_name']
                baseline_best_accuracy = results['best_metrics']['Accuracy']
                
                print(f"\n[Candidate 1] Baseline Model (Optimized):")
                print(f"   Model: {baseline_best_name}")
                print(f"   Accuracy: {baseline_best_accuracy:.4f}")
                print(f"   Features: {len(X.columns)}")
                
                print(f"\n[Candidate 2] Improved Model (Optimized + FE):")
                print(f"   Model: {best_improved_name}")
                print(f"   Accuracy: {best_improved_acc:.4f}")
                print(f"   Features: {len(X_train_df.columns)}")
                
                # AUTO-SELECT: Choose the better one
                improvement = best_improved_acc - baseline_best_accuracy
                
                if improvement > 0:
                    print(f"\n WINNER: Improved Model")
                    print(f"   Selected: {best_improved_name}")
                    print(f"   Accuracy: {best_improved_acc:.4f}")
                    print(f"   Improvement: +{improvement:.4f} (+{improvement/baseline_best_accuracy*100:.2f}%)")
                    
                    # Update results with improved model
                    results['best_model'] = best_improved_model
                    results['best_model_name'] = best_improved_name
                    results['best_metrics']['Accuracy'] = best_improved_acc
                    results['improved_selected'] = True
                    
                    # CRITICAL FIX: Ensure pipeline consistency
                    # Store transformer and updated data so API can use them
                    results['feature_transformer'] = fe_engine
                    results['X_train_scaled'] = X_train_improved
                    results['X_test_scaled'] = X_test_improved
                    # Update feature names for XAI
                    if hasattr(fe_engine, 'get_feature_names'):
                         results['feature_names'] = fe_engine.get_feature_names()
                    else:
                         # Fallback
                         results['feature_names'] = X.columns.tolist() + new_features
                         
                    # RE-RUN XAI FOR THE FINAL IMPROVED MODEL
                    # This ensures Web UI shows SHAP/PI for the model actually being used
                    print("\n[Step 4/3] Updating XAI Analysis for Final Improved Model...")
                    try:
                        final_xai = XAIExplainer(
                            model=best_improved_model,
                            X_train=X_train_improved,
                            X_test=X_test_improved,
                            y_train=y_train,
                            y_test=y_test,
                            X_train_scaled=X_train_improved, # Already scaled/transformed
                            X_test_scaled=X_test_improved,
                            feature_names=results['feature_names']
                        )
                        # We only need Global explanations updated in results
                        final_xai.calculate_shap()
                        final_shap_imp = final_xai.get_shap_importance()
                        final_shap_summary = final_xai.get_shap_summary_data()
                        
                        # Permutation importance for new model
                        from sklearn.inspection import permutation_importance
                        print("    Calculating Permutation Importance for Improved Model...")
                        perm_results = permutation_importance(
                            best_improved_model, X_test_improved, y_test, 
                            n_repeats=10, random_state=42, n_jobs=-1
                        )
                        final_perm_imp = pd.DataFrame({
                            'Feature': results['feature_names'],
                            'Importance': perm_results.importances_mean,
                            'Std': perm_results.importances_std
                        }).sort_values('Importance', ascending=False)
                        
                        # Update results['xai_results'] with FINAL model insights
                        results['xai_results']['shap_importance'] = final_shap_imp
                        results['xai_results']['shap_summary_data'] = final_shap_summary
                        results['xai_results']['permutation_importance'] = final_perm_imp
                        print("    XAI Results Updated to reflect Final Model.")
                        
                    except Exception as e:
                        print(f"    Warning: Could not update XAI for improved model: {e}")
                        
                else:
                    print(f"\n WINNER: Baseline Model")
                    print(f"   Selected: {baseline_best_name}")
                    print(f"   Accuracy: {baseline_best_accuracy:.4f}")
                    print(f"   Note: Feature Engineering did not improve performance")
                    results['improved_selected'] = False
                
                # Store results with correct key name
                results['improvement_results'] = {
                    'improved_results_df': improved_results_df,
                    'comparison_df': comparison_df
                }
                results['improved_results'] = improved_results_df
                results['improvement_comparison'] = comparison_df
                results['new_features'] = new_features
                results['xai_results'] = xai_results
                
                # Generate explanation
                print("\n" + "="*60)
                print(" IMPROVEMENT EXPLANATION")
                print("="*60)
                print()
                
                if improvement > 0:
                    print(f" **Result: PERFORMANCE IMPROVED**")
                    print(f"   - Accuracy increased: +{improvement:.4f} (+{improvement/baseline_best_accuracy*100:.2f}%)")
                    print(f"   - Added {len(new_features)} interaction features")
                    print()
                    print(f" **Strategy Applied:**")
                    print(f"   1. **XAI Analysis**: Identified top {len(top_features)} important features")
                    print(f"   2. **Feature Engineering**: Created {len(new_features)} interaction features")
                    print(f"   3. **Model Re-training**: Utilized enriched feature space")
                    print()
                    print(f"💡 **Recommendation**: USE IMPROVED MODEL for production")
                    print()
                    print(f" **Interaction Features Created:**")
                    for i, feat in enumerate(new_features[:5], 1):
                        print(f"   {i}. {feat}")
                    if len(new_features) > 5:
                        print(f"   (and {len(new_features) - 5} more...)")
                    
                    results['improvement_explanation'] = f"Improved by {improvement/baseline_best_accuracy*100:.2f}% through XAI-guided feature engineering"
                else:
                    print(f"➖ **Result: NO SIGNIFICANT IMPROVEMENT**")
                    print(f"   - Accuracy change: {improvement:+.4f} ({improvement/baseline_best_accuracy*100:+.2f}%)")
                    print(f"   - Added {len(new_features)} interaction features but no gain")
                    print()
                    print(f" **Possible Reasons:**")
                    print(f"   - Dataset may already be at optimal performance")
                    print(f"   - Interaction features did not capture additional patterns")
                    print(f"   - Model may have reached its capacity limit")
                    print()
                    print(f"💡 **Recommendation**: USE BASELINE MODEL (simpler, same performance)")
                    
                    results['improvement_explanation'] = f"Feature Engineering tested but provided no improvement ({improvement/baseline_best_accuracy*100:+.2f}%)"
                
                print("\n XAI-Guided Improvement Complete!")
                
            except Exception as e:
                print(f"\n⚠️  XAI-Guided Improvement failed: {str(e)}")
                print("   Continuing with baseline results...")
                import traceback
                traceback.print_exc()
        
        return results


if __name__ == "__main__":
    # Test the trainer
    from data_processor import DataProcessor
    
    processor = DataProcessor()
    data, stats = processor.process_pipeline('heart_statlog_cleveland_hungary_final.csv')
    X, y = processor.get_X_y(data)
    
    trainer = ModelTrainer()
    results = trainer.full_training_pipeline(X, y)
    
    print("\nTraining completed!")
    print(f"Best model: {results['best_model_name']}")
