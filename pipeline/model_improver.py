"""
XAI-Guided Model Improvement
Automatically improves models based on XAI insights
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, List


class ModelImprover:
    """
    Improves models based on XAI analysis results
    
    Strategies:
    1. Feature Selection: Remove low-importance features
    2. Feature Engineering: Create interaction features from top important features
    """
    
    def __init__(self, xai_results: Dict, threshold: float = 0.01):
        """
        Args:
            xai_results: Results from XAIExplainer
            threshold: Importance threshold for feature selection
        """
        self.xai_results = xai_results
        self.threshold = threshold
        self.improvement_report = {}
    
    def extract_insights(self) -> Dict:
        """Extract actionable insights from XAI results"""
        
        # Get feature importance from permutation importance
        perm_importance = self.xai_results['permutation_importance']
        
        # Top features (importance > threshold)
        important_features = perm_importance[
            perm_importance['Importance'] > self.threshold
        ]['Feature'].tolist()
        
        # Low importance features (candidates for removal)
        low_importance = perm_importance[
            perm_importance['Importance'] <= self.threshold
        ]['Feature'].tolist()
        
        # Top 5 most important features for interactions
        top_5_features = perm_importance.nlargest(5, 'Importance')['Feature'].tolist()
        
        insights = {
            'important_features': important_features,
            'low_importance_features': low_importance,
            'top_features_for_interaction': top_5_features,
            'n_features_before': len(perm_importance),
            'n_features_after': len(important_features)
        }
        
        print(f"\n XAI Insights Extracted:")
        print(f"   - Features to keep: {len(important_features)}")
        print(f"   - Features to remove: {len(low_importance)}")
        print(f"   - Top features for interactions: {top_5_features[:3]}")
        
        return insights
    
    def feature_selection_improvement(
        self, 
        models: Dict, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Strategy 1: Remove low-importance features
        
        Returns:
            improved_models: Retrained models
            X_train_improved: Training data with selected features
            X_test_improved: Test data with selected features
            report: Improvement statistics
        """
        print("\n🔧 Applying Feature Selection Strategy...")
        
        insights = self.extract_insights()
        important_features = insights['important_features']
        
        if len(important_features) == 0:
            print("⚠️  No features meet the importance threshold. Skipping feature selection.")
            return models, X_train, X_test, {'strategy': 'Feature Selection', 'skipped': True}
        
        # Get all feature names from XAI results
        all_features = self.xai_results['permutation_importance']['Feature'].tolist()
        
        # Handle both DataFrame and numpy array inputs
        if isinstance(X_train, pd.DataFrame):
            # DataFrame: use column names directly
            X_train_improved = X_train[important_features].copy()
            X_test_improved = X_test[important_features].copy()
        else:
            # Numpy array: convert feature names to indices
            feature_indices = [all_features.index(feat) for feat in important_features]
            X_train_improved = X_train[:, feature_indices].copy()
            X_test_improved = X_test[:, feature_indices].copy()
        
        print(f"   Removed {len(insights['low_importance_features'])} low-importance features")
        print(f"   Training with {len(important_features)} features")
        
        # Retrain models
        improved_models = {}
        for name, model in models.items():
            # Clone model to avoid modifying original
            new_model = clone(model)
            new_model.fit(X_train_improved, y_train)
            improved_models[name] = new_model
        
        # Generate report
        report = {
            'strategy': 'Feature Selection',
            'features_removed': insights['low_importance_features'],
            'n_features_before': insights['n_features_before'],
            'n_features_after': insights['n_features_after'],
            'features_kept': important_features
        }
        
        print(" Feature Selection completed")
        
        return improved_models, X_train_improved, X_test_improved, report
    
    def feature_engineering_improvement(
        self,
        models: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Strategy 2: Create interaction features from top important features
        
        Returns:
            improved_models: Retrained models
            X_train_improved: Training data with interaction features
            X_test_improved: Test data with interaction features
            report: Improvement statistics
        """
        print("\n🔧 Applying Feature Engineering Strategy...")
        
        insights = self.extract_insights()
        top_features = insights['top_features_for_interaction']
        
        # Get all feature names from XAI results
        all_features = self.xai_results['permutation_importance']['Feature'].tolist()
        
        # Handle numpy arrays vs DataFrames
        if isinstance(X_train, pd.DataFrame):
            # DataFrame: work directly with column names
            X_train_improved = X_train.copy()
            X_test_improved = X_test.copy()
            
            new_features = []
            
            # Create pairwise interactions for top features
            for i in range(len(top_features)):
                for j in range(i + 1, len(top_features)):
                    feat1, feat2 = top_features[i], top_features[j]
                    
                    # Check if features exist in dataframe
                    if feat1 not in X_train.columns or feat2 not in X_train.columns:
                        continue
                    
                    # Multiplication interaction
                    interaction_name = f"{feat1}_x_{feat2}"
                    X_train_improved[interaction_name] = X_train[feat1] * X_train[feat2]
                    X_test_improved[interaction_name] = X_test[feat1] * X_test[feat2]
                    new_features.append(interaction_name)
        else:
            # Numpy array: convert to DataFrame temporarily
            X_train_df = pd.DataFrame(X_train, columns=all_features)
            X_test_df = pd.DataFrame(X_test, columns=all_features)
            
            new_features = []
            
            # Create pairwise interactions for top features
            for i in range(len(top_features)):
                for j in range(i + 1, len(top_features)):
                    feat1, feat2 = top_features[i], top_features[j]
                    
                    # Check if features exist
                    if feat1 not in all_features or feat2 not in all_features:
                        continue
                    
                    # Multiplication interaction
                    interaction_name = f"{feat1}_x_{feat2}"
                    X_train_df[interaction_name] = X_train_df[feat1] * X_train_df[feat2]
                    X_test_df[interaction_name] = X_test_df[feat1] * X_test_df[feat2]
                    new_features.append(interaction_name)
            
            # Convert back to numpy arrays
            X_train_improved = X_train_df.values
            X_test_improved = X_test_df.values
        
        print(f"   Created {len(new_features)} interaction features")
        if isinstance(X_train_improved, pd.DataFrame):
            print(f"   Training with {len(X_train_improved.columns)} total features")
        else:
            print(f"   Training with {X_train_improved.shape[1]} total features")
        
        # Retrain models
        improved_models = {}
        for name, model in models.items():
            new_model = clone(model)
            new_model.fit(X_train_improved, y_train)
            improved_models[name] = new_model
        
        # Generate report
        report = {
            'strategy': 'Feature Engineering',
            'new_features': new_features,
            'n_features_added': len(new_features),
            'base_features_used': top_features,
            'n_features_total': len(X_train_improved.columns)
        }
        
        print(" Feature Engineering completed")
        
        return improved_models, X_train_improved, X_test_improved, report
    
    def evaluate_improvements(
        self,
        baseline_models: Dict,
        improved_models: Dict,
        X_test: pd.DataFrame,
        X_test_improved: pd.DataFrame,
        y_test: pd.Series,
        strategy_name: str
    ) -> pd.DataFrame:
        """
        Compare baseline vs improved models
        
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for model_name in baseline_models.keys():
            # Baseline predictions
            baseline_pred = baseline_models[model_name].predict(X_test)
            baseline_acc = accuracy_score(y_test, baseline_pred)
            
            # Improved predictions
            improved_pred = improved_models[model_name].predict(X_test_improved)
            improved_acc = accuracy_score(y_test, improved_pred)
            
            # Calculate improvement
            improvement = improved_acc - baseline_acc
            improvement_pct = (improvement / baseline_acc) * 100
            
            results.append({
                'Model': model_name,
                'Baseline Accuracy': f"{baseline_acc:.4f}",
                'Improved Accuracy': f"{improved_acc:.4f}",
                'Improvement': f"{improvement:+.4f}",
                'Improvement %': f"{improvement_pct:+.2f}%",
                'Strategy': strategy_name
            })
        
        return pd.DataFrame(results)
    
    def auto_improve(
        self,
        models: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        strategy: str = 'feature_selection'
    ) -> Dict:
        """
        Automatically improve models based on XAI insights
        
        Args:
            strategy: 'feature_selection', 'feature_engineering', or 'both'
        
        Returns:
            results: Dictionary containing improved models and reports
        """
        print("\n" + "="*60)
        print(" AUTOMATIC MODEL IMPROVEMENT (XAI-Guided)")
        print("="*60)
        
        results = {}
        
        if strategy in ['feature_selection', 'both']:
            improved_models, X_train_imp, X_test_imp, report = \
                self.feature_selection_improvement(models, X_train, y_train, X_test, y_test)
            
            # Evaluate improvements
            comparison = self.evaluate_improvements(
                models, improved_models, X_test, X_test_imp, y_test, 'Feature Selection'
            )
            
            results['feature_selection'] = {
                'models': improved_models,
                'X_train': X_train_imp,
                'X_test': X_test_imp,
                'report': report,
                'comparison': comparison
            }
        
        if strategy in ['feature_engineering', 'both']:
            improved_models, X_train_imp, X_test_imp, report = \
                self.feature_engineering_improvement(models, X_train, y_train, X_test, y_test)
            
            # Evaluate improvements
            comparison = self.evaluate_improvements(
                models, improved_models, X_test, X_test_imp, y_test, 'Feature Engineering'
            )
            
            results['feature_engineering'] = {
                'models': improved_models,
                'X_train': X_train_imp,
                'X_test': X_test_imp,
                'report': report,
                'comparison': comparison
            }
        
        print("\n Model Improvement Pipeline Completed!")
        print("="*60)
        
        return results


class InteractionFeatureEngine:
    """
    Handles Feature Engineering (Interactions) effectively for pipeline consistency.
    Stores the logic to reproduce features during inference.
    """
    
    def __init__(self, top_features_count=5):
        self.top_features_count = top_features_count
        self.interactions = [] # List of tuples (feat1, feat2)
        self.new_feature_names = []
        self.original_feature_names = []
        
    def fit(self, X_df, xai_results):
        """
        Identify top features and determine interactions to create
        """
        self.original_feature_names = X_df.columns.tolist()
        
        # Extract top features from XAI
        perm_importance = xai_results['permutation_importance']
        top_features = perm_importance.nlargest(self.top_features_count, 'Importance')['Feature'].tolist()
        
        # Generate combinations
        from itertools import combinations
        self.interactions = list(combinations(top_features, 2))[:5]
        
        self.new_feature_names = []
        for feat1, feat2 in self.interactions:
            self.new_feature_names.append(f"{feat1}_x_{feat2}")
            
        print(f"InteractionFeatureEngine fitted. Created {len(self.interactions)} interactions.")
        return self

    def transform(self, X_df):
        """
        Apply the learned interactions to new data
        """
        X_new = X_df.copy()
        
        # Verify columns exist
        missing_cols = [col for col in self.original_feature_names if col not in X_new.columns]
        if missing_cols:
             # Try to handle if X_new is just numpy array converted to DF without correct columns
             if len(X_new.columns) == len(self.original_feature_names):
                 X_new.columns = self.original_feature_names
        
        for (feat1, feat2), new_name in zip(self.interactions, self.new_feature_names):
            if feat1 in X_new.columns and feat2 in X_new.columns:
                X_new[new_name] = X_new[feat1] * X_new[feat2]
            else:
                # Fallback if features missing
                X_new[new_name] = 0
                
        return X_new
    
    def get_feature_names(self):
        return self.original_feature_names + self.new_feature_names
