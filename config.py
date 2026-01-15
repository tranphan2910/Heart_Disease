import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration - Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL = "openai/gpt-oss-120b"  # GPT-OSS 120B reasoning model
GROQ_REASONING_FORMAT = "hidden"  # Options: "parsed", "raw", "hidden"
GROQ_INCLUDE_REASONING = True  # Include reasoning in response

# LLM Rate Limiting Configuration
GROQ_REQUEST_DELAY = 1.5  # Seconds between API calls to avoid rate limiting
GROQ_MAX_RETRIES = 5  # Maximum retry attempts on 429 errors
GROQ_LANGUAGE = "english"  # Language for LLM interpretations: "english" or "vietnamese"

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature Configuration
NUMERICAL_FEATURES = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 
                       'exercise angina', 'ST slope']
ONE_HOT_FEATURES = ['chest pain type', 'resting ecg', 'ST slope']
DROP_FEATURES = ['AgeBand', 'cholesterol', 'fasting blood sugar', 
                'chest pain type_1', 'resting ecg_1', 'ST slope_0', 'ST slope_3']

# Model Hyperparameter Grids
MODEL_PARAMS = {
    "MLP": {
        "hidden_layer_sizes": [(100,), (100, 50)],
        "activation": ['relu', 'tanh'],
        "alpha": [0.0001, 0.001],
        "learning_rate": ['constant', 'adaptive'],
        "solver": ['adam'],
        "max_iter": [500]
    },
    "GaussianNB": {
        "var_smoothing": [1e-9, 1e-8, 1e-7]
    },
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ['lbfgs'],
        "penalty": ['l2'],
        "max_iter": [500]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    },
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0]
    }
}

# Flask Configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Streamlit Configuration
STREAMLIT_THEME = "light"
PAGE_TITLE = "Heart Disease Prediction - XAI Analysis"
PAGE_ICON = "❤️"
