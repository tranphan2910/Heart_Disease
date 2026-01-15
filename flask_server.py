"""
Flask API Server để xử lý requests từ Streamlit UI
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os
from pipeline import DataProcessor, ModelTrainer, XAIExplainer
from utils import LLMInterpreter

app = Flask(__name__)
CORS(app)  # Enable CORS for Streamlit

# Global variables để cache models
_cached_model = None
_cached_scaler = None
_cached_feature_names = None
_cached_training_data = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Flask server is running"})


@app.route('/train', methods=['POST'])
def train_model():
    """
    Train model endpoint
    Expected JSON: {
        "data_path": "path/to/data.csv"
    }
    """
    try:
        data = request.get_json()
        data_path = data.get('data_path')
        
        if not data_path or not os.path.exists(data_path):
            return jsonify({"error": "Invalid data path"}), 400
        
        # 1. Process data
        processor = DataProcessor()
        processed_data, stats = processor.process_pipeline(data_path)
        X, y = processor.get_X_y(processed_data)
        
        # 2. Train models
        trainer = ModelTrainer()
        training_results = trainer.full_training_pipeline(X, y)
        
        # Cache results
        global _cached_model, _cached_scaler, _cached_feature_names, _cached_training_data
        _cached_model = training_results['best_model']
        _cached_scaler = trainer.scaler
        _cached_feature_names = X.columns.tolist()
        _cached_training_data = training_results
        
        # Prepare response
        response = {
            "status": "success",
            "best_model": training_results['best_model_name'],
            "metrics": {
                "accuracy": float(training_results['best_metrics']['Accuracy']),
                "precision": float(training_results['best_metrics']['Precision']),
                "recall": float(training_results['best_metrics']['Recall']),
                "f1_score": float(training_results['best_metrics']['F1 Score'])
            },
            "data_stats": {
                "original_shape": stats['original_shape'],
                "final_shape": stats['final_shape'],
                "num_features": len(stats['final_features'])
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Expected JSON: {
        "features": {...}  # Dictionary of feature values
    }
    """
    try:
        if _cached_model is None:
            return jsonify({"error": "Model not trained yet. Call /train first."}), 400
        
        data = request.get_json()
        features = data.get('features')
        
        if not features:
            return jsonify({"error": "No features provided"}), 400
        
        # Convert to DataFrame and scale
        feature_df = pd.DataFrame([features])
        feature_scaled = _cached_scaler.transform(feature_df)
        
        # Predict
        prediction = _cached_model.predict(feature_scaled)[0]
        prediction_proba = _cached_model.predict_proba(feature_scaled)[0]
        
        response = {
            "prediction": int(prediction),
            "prediction_label": "Heart Disease" if prediction == 1 else "No Heart Disease",
            "probability": {
                "no_disease": float(prediction_proba[0]),
                "disease": float(prediction_proba[1])
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/xai/analyze', methods=['POST'])
def xai_analyze():
    """
    XAI Analysis endpoint
    Returns SHAP, LIME, and Permutation Importance results
    """
    try:
        if _cached_training_data is None:
            return jsonify({"error": "Model not trained yet. Call /train first."}), 400
        
        # Run XAI analysis
        explainer = XAIExplainer(
            model=_cached_training_data['best_model'],
            X_train=_cached_training_data['X_train'],
            X_test=_cached_training_data['X_test'],
            y_train=_cached_training_data['y_train'],
            y_test=_cached_training_data['y_test'],
            X_train_scaled=_cached_training_data['X_train_scaled'],
            X_test_scaled=_cached_training_data['X_test_scaled'],
            feature_names=_cached_feature_names
        )
        
        xai_results = explainer.full_xai_pipeline()
        
        # Convert to JSON-serializable format
        response = {
            "shap_importance": xai_results['shap_importance'].to_dict('records'),
            "permutation_importance": xai_results['permutation_importance'].to_dict('records'),
            "lime_sample_count": len(xai_results['lime_explanations'])
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/xai/interpret', methods=['POST'])
def xai_interpret():
    """
    LLM Interpretation endpoint
    Expected JSON: {
        "xai_results": {...},
        "model_info": {...}
    }
    """
    try:
        data = request.get_json()
        xai_results = data.get('xai_results')
        model_info = data.get('model_info')
        
        if not xai_results or not model_info:
            return jsonify({"error": "Missing xai_results or model_info"}), 400
        
        # Initialize LLM interpreter
        interpreter = LLMInterpreter()
        
        # Convert back to DataFrame format
        shap_df = pd.DataFrame(xai_results['shap_importance'])
        perm_df = pd.DataFrame(xai_results['permutation_importance'])
        
        # Generate interpretations
        interpretations = {
            "shap": interpreter.interpret_shap_importance(shap_df, model_info),
            "permutation": interpreter.interpret_permutation_importance(perm_df),
            "comparison": interpreter.compare_methods(shap_df, perm_df)
        }
        
        return jsonify(interpretations)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/models/list', methods=['GET'])
def list_models():
    """List all trained models and their performance"""
    try:
        if _cached_training_data is None:
            return jsonify({"error": "No models trained yet"}), 400
        
        tuned_results = _cached_training_data['tuned_results']
        
        models_list = []
        for _, row in tuned_results.iterrows():
            models_list.append({
                "name": row['Model Name'],
                "accuracy": float(row['Accuracy']),
                "precision": float(row['Precision']),
                "recall": float(row['Recall']),
                "f1_score": float(row['F1 Score'])
            })
        
        return jsonify({"models": models_list})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import config
    
    print("=" * 60)
    print("Starting Flask API Server for Heart Disease Prediction")
    print("=" * 60)
    print(f"Server will run on http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print("\nAvailable endpoints:")
    print("  GET  /health              - Health check")
    print("  POST /train               - Train models")
    print("  POST /predict             - Make predictions")
    print("  POST /xai/analyze         - Run XAI analysis")
    print("  POST /xai/interpret       - Get LLM interpretations")
    print("  GET  /models/list         - List all models")
    print("=" * 60)
    
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
