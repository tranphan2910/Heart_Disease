import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from eli5.sklearn import PermutationImportance
import warnings
warnings.filterwarnings('ignore')


class XAIExplainer:
    """
    XAI (Explainable AI) với SHAP, LIME, và Permutation Importance
    """
    
    def __init__(self, model, X_train, X_test, y_train, y_test, 
                 X_train_scaled, X_test_scaled, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.feature_names = feature_names
        
        # Initialize explainers
        self.shap_explainer = None
        self.shap_values = None
        self.lime_explainer = None
        self.perm_importance = None
    
    def calculate_shap(self):
        """Tính SHAP values"""
        print("Calculating SHAP values...")
        
        try:
            # Use TreeExplainer for tree-based models
            if hasattr(self.model, 'tree_') or hasattr(self.model, 'estimators_'):
                self.shap_explainer = shap.TreeExplainer(self.model, self.X_train_scaled)
            else:
                # Use KernelExplainer for other models (slower)
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    shap.sample(self.X_train_scaled, 100)
                )
            
            self.shap_values = self.shap_explainer.shap_values(
                self.X_test_scaled, 
                check_additivity=False
            )
            
            print(f"SHAP values shape: {np.array(self.shap_values).shape}")
            return self.shap_values
            
        except Exception as e:
            print(f"Error calculating SHAP: {e}")
            return None
    
    def get_shap_importance(self):
        """Lấy feature importance từ SHAP"""
        if self.shap_values is None:
            self.calculate_shap()
        
        # Handle multi-output SHAP values (for binary classification)
        if isinstance(self.shap_values, list) or len(self.shap_values.shape) == 3:
            # Use class 1 (positive class)
            shap_vals = self.shap_values[:, :, 1] if len(self.shap_values.shape) == 3 else self.shap_values[1]
        else:
            shap_vals = self.shap_values
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP Importance': mean_abs_shap
        }).sort_values('SHAP Importance', ascending=False)
        
        return importance_df
    
    def get_shap_summary_data(self):
        """Chuẩn bị data cho SHAP summary plot"""
        if self.shap_values is None:
            self.calculate_shap()
        
        # Get SHAP values for class 1
        if isinstance(self.shap_values, list) or len(self.shap_values.shape) == 3:
            shap_vals = self.shap_values[:, :, 1] if len(self.shap_values.shape) == 3 else self.shap_values[1]
        else:
            shap_vals = self.shap_values
        
        return {
            'shap_values': shap_vals,
            'features': self.X_test_scaled,
            'feature_names': self.feature_names
        }
    
    def setup_lime(self):
        """Khởi tạo LIME explainer"""
        print("Setting up LIME explainer...")
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.feature_names,
            class_names=['No Disease', 'Disease'],
            mode='classification'
        )
        
        return self.lime_explainer
    
    def explain_instance_lime(self, instance_idx, num_features=10):
        """Giải thích một instance cụ thể bằng LIME"""
        if self.lime_explainer is None:
            self.setup_lime()
        
        instance = self.X_test_scaled[instance_idx]
        
        # Prediction function
        predict_fn = lambda x: self.model.predict_proba(x)
        
        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features
        )
        
        # Extract explanation data
        exp_list = explanation.as_list()
        exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Weight'])
        
        # Get prediction
        prediction = self.model.predict(instance.reshape(1, -1))[0]
        prediction_proba = self.model.predict_proba(instance.reshape(1, -1))[0]
        
        return {
            'explanation': explanation,
            'explanation_df': exp_df,
            'prediction': prediction,
            'prediction_proba': prediction_proba,
            'instance_idx': instance_idx
        }
    
    def explain_multiple_instances(self, num_instances=5):
        """Giải thích nhiều instances"""
        explanations = []
        
        # Get some correct and incorrect predictions
        predictions = self.model.predict(self.X_test_scaled)
        correct_idx = np.where(predictions == self.y_test.values)[0]
        incorrect_idx = np.where(predictions != self.y_test.values)[0]
        
        # Mix correct and incorrect
        selected_idx = []
        if len(correct_idx) > 0:
            selected_idx.extend(np.random.choice(correct_idx, min(num_instances//2, len(correct_idx)), replace=False))
        if len(incorrect_idx) > 0:
            selected_idx.extend(np.random.choice(incorrect_idx, min(num_instances//2, len(incorrect_idx)), replace=False))
        
        for idx in selected_idx:
            exp = self.explain_instance_lime(idx)
            exp['actual'] = self.y_test.iloc[idx]
            exp['correct'] = exp['prediction'] == exp['actual']
            explanations.append(exp)
        
        return explanations
    
    def calculate_permutation_importance(self, n_repeats=10):
        """Tính Permutation Importance"""
        print("Calculating Permutation Importance...")
        
        self.perm_importance = PermutationImportance(
            self.model, 
            random_state=42,
            n_iter=n_repeats
        ).fit(self.X_test_scaled, self.y_test)
        
        # Extract importance scores
        importances = self.perm_importance.feature_importances_
        std = self.perm_importance.feature_importances_std_
        
        perm_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances,
            'Std': std
        }).sort_values('Importance', ascending=False)
        
        return perm_df
    
    def full_xai_pipeline(self):
        """
        Pipeline XAI hoàn chỉnh: SHAP + LIME + Permutation Importance
        
        Returns:
            xai_results: Dictionary chứa tất cả kết quả XAI
        """
        print("\n" + "=" * 60)
        print("Starting XAI Analysis Pipeline...")
        print("=" * 60)
        
        results = {}
        
        # 1. SHAP Analysis
        print("\n[1/3] SHAP Analysis...")
        self.calculate_shap()
        shap_importance = self.get_shap_importance()
        shap_summary_data = self.get_shap_summary_data()
        results['shap_importance'] = shap_importance
        results['shap_summary_data'] = shap_summary_data
        results['shap_values'] = self.shap_values
        print(f"  Top 3 features by SHAP: {shap_importance.head(3)['Feature'].tolist()}")
        
        # 2. LIME Analysis
        print("\n[2/3] LIME Analysis...")
        lime_explanations = self.explain_multiple_instances(num_instances=30)  # Tăng lên 30 instances
        results['lime_explanations'] = lime_explanations
        print(f"  Explained {len(lime_explanations)} instances")
        
        # 3. Permutation Importance
        print("\n[3/3] Permutation Importance...")
        perm_importance = self.calculate_permutation_importance()
        results['permutation_importance'] = perm_importance
        print(f"  Top 3 features by Perm. Importance: {perm_importance.head(3)['Feature'].tolist()}")
        
        print("\n" + "=" * 60)
        print("XAI Analysis Completed!")
        print("=" * 60)
        
        return results
    
    def threshold_analysis(self, baseline_accuracy, thresholds=[0.001, 0.005, 0.010, 0.020, 0.050]):
        """
        Analyze impact of removing features below importance threshold
        
        Args:
            baseline_accuracy: Baseline model accuracy (with all features)
            thresholds: List of importance thresholds to test
            
        Returns:
            DataFrame with threshold analysis results
        """
        print("\n" + "=" * 60)
        print("Threshold Analysis - Feature Removal Impact")
        print("=" * 60)
        
        from sklearn.metrics import accuracy_score
        
        # Get feature importance from permutation importance
        if self.perm_importance is None:
            self.calculate_permutation_importance()
        
        perm_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.perm_importance.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        results = []
        
        for threshold in thresholds:
            # Find features to remove (importance < threshold)
            features_to_remove = perm_df[perm_df['Importance'] < threshold]['Feature'].tolist()
            num_removed = len(features_to_remove)
            
            if num_removed == 0:
                # No features removed
                results.append({
                    'Threshold': threshold,
                    'Features Removed': 0,
                    'Best Accuracy': baseline_accuracy,
                    'Change': '0.00%',
                    'Status': 'No change'
                })
                continue
            
            # Get indices of features to keep
            feature_indices_to_keep = [i for i, feat in enumerate(self.feature_names) 
                                       if feat not in features_to_remove]
            
            if len(feature_indices_to_keep) == 0:
                # All features removed - skip
                results.append({
                    'Threshold': threshold,
                    'Features Removed': num_removed,
                    'Best Accuracy': 0.0,
                    'Change': '-100.00%',
                    'Status': 'Worse '
                })
                continue
            
            # Train model with reduced features
            X_train_reduced = self.X_train_scaled[:, feature_indices_to_keep]
            X_test_reduced = self.X_test_scaled[:, feature_indices_to_keep]
            
            try:
                # Clone and retrain model
                from sklearn.base import clone
                reduced_model = clone(self.model)
                reduced_model.fit(X_train_reduced, self.y_train)
                
                # Evaluate
                y_pred = reduced_model.predict(X_test_reduced)
                accuracy = accuracy_score(self.y_test, y_pred)
                
                # Calculate change
                change = accuracy - baseline_accuracy
                change_pct = (change / baseline_accuracy) * 100
                
                # Determine status
                if change > 0:
                    status = 'Better '
                elif change < 0:
                    status = 'Worse '
                else:
                    status = 'No change'
                
                results.append({
                    'Threshold': threshold,
                    'Features Removed': num_removed,
                    'Best Accuracy': f"{accuracy:.2%}",
                    'Change': f"{change_pct:+.2f}%",
                    'Status': status
                })
                
                print(f"Threshold {threshold}: Removed {num_removed} features → Accuracy: {accuracy:.2%} ({change_pct:+.2f}%)")
                
            except Exception as e:
                print(f"Error at threshold {threshold}: {str(e)}")
                results.append({
                    'Threshold': threshold,
                    'Features Removed': num_removed,
                    'Best Accuracy': 'Error',
                    'Change': 'N/A',
                    'Status': 'Error'
                })
        
        results_df = pd.DataFrame(results)
        
        print("\n" + "=" * 60)
        print(f"Baseline Accuracy: {baseline_accuracy:.2%}")
        print("=" * 60)
        
        return results_df


if __name__ == "__main__":
    # Test XAI explainer
    from data_processor import DataProcessor
    from model_trainer import ModelTrainer
    
    processor = DataProcessor()
    data, stats = processor.process_pipeline('heart_statlog_cleveland_hungary_final.csv')
    X, y = processor.get_X_y(data)
    
    trainer = ModelTrainer()
    training_results = trainer.full_training_pipeline(X, y)
    
    explainer = XAIExplainer(
        model=training_results['best_model'],
        X_train=training_results['X_train'],
        X_test=training_results['X_test'],
        y_train=training_results['y_train'],
        y_test=training_results['y_test'],
        X_train_scaled=training_results['X_train_scaled'],
        X_test_scaled=training_results['X_test_scaled'],
        feature_names=X.columns.tolist()
    )
    
    xai_results = explainer.full_xai_pipeline()
    print("\nXAI analysis completed!")
