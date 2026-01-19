import os
import json
import requests
import time
import numpy as np
from typing import Dict, List, Any
from functools import wraps
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import config sau khi load env
import config


class LLMInterpreter:
    """
    Sử dụng Groq API (GPT-OSS-120B) để tự động giải thích kết quả XAI
    Có tích hợp retry logic và rate limiting để tránh lỗi 429
    """
    
    # Medical context cho features
    FEATURE_MEDICAL_CONTEXT = {
        'ST slope_1': 'Độ dốc đoạn ST lên (↑) - tim đáp ứng tốt với vận động, ít nguy cơ thiếu máu cơ tim',
        'ST slope_2': 'Độ dốc đoạn ST ngang (↔) - dấu hiệu cảnh báo thiếu máu cơ tim',
        'ST slope_0': 'Độ dốc đoạn ST xuống (↓) - nguy cơ cao thiếu máu nghiêm trọng',
        'chest pain type_0': 'Đau thắt ngực điển hình (typical angina) - đau khi gắng sức',
        'chest pain type_2': 'Đau ngực không điển hình (atypical angina)',
        'chest pain type_3': 'Đau ngực không do tim (non-anginal pain)',
        'chest pain type_4': 'Không có triệu chứng đau (asymptomatic) - "silent ischemia", khó phát hiện sớm',
        'oldpeak': 'Độ giảm ST sau vận động (mmHg) - càng cao càng thiếu máu nghiêm trọng',
        'exercise angina': 'Đau ngực khi tập (1=có, 0=không) - dấu hiệu trực tiếp bệnh động mạch vành',
        'sex': 'Giới tính (0=nữ, 1=nam) - nam có tỷ lệ mắc cao hơn',
        'age': 'Tuổi - nguy cơ tăng sau 45 tuổi (nam) và 55 tuổi (nữ)',
        'max heart rate': 'Nhịp tim tối đa (bpm) - thấp hơn dự đoán → sức khỏe tim kém',
        'resting bp s': 'Huyết áp nghỉ (mmHg) - cao kéo dài gây xơ vữa động mạch',
        'resting ecg_0': 'ECG nghỉ bình thường',
        'resting ecg_1': 'ECG nghỉ có bất thường sóng ST-T - dấu hiệu sớm thiếu máu',
        'resting ecg_2': 'ECG nghỉ phì đại thất trái - hậu quả tăng huyết áp lâu dài',
        'cholesterol': 'Cholesterol (mg/dl) - cao gây xơ vữa mạch máu',
        'fasting blood sugar': 'Đường huyết lúc đói >120mg/dl - liên quan đến đái tháo đường'
    }
    
    def __init__(self, api_key: str = None, api_base: str = None, model: str = None, 
                 request_delay: float = None, max_retries: int = None):
        """
        Initialize LLM Interpreter with Groq API
        
        Args:
            api_key: Groq API key (nếu None sẽ lấy từ env)
            api_base: Groq API base URL
            model: Model name (default: openai/gpt-oss-120b)
            request_delay: Delay giữa các API calls (seconds) để tránh rate limit
            max_retries: Số lần retry tối đa khi gặp lỗi 429
        """
        # Fallback chain: parameter → config → environment variable
        self.api_key = api_key or config.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        self.api_base = api_base or config.GROQ_API_BASE or os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        self.model = model or config.GROQ_MODEL or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.request_delay = request_delay if request_delay is not None else config.GROQ_REQUEST_DELAY
        self.max_retries = max_retries if max_retries is not None else config.GROQ_MAX_RETRIES
        self.last_request_time = 0
        
        if not self.api_key:
            print("Warning: GROQ_API_KEY not set. LLM interpretation will not work.")
            print("Please set it in .env file or pass it as parameter.")
    
    def call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """
        Public wrapper for _call_groq_api to allow direct prompts
        """
        return self._call_groq_api(prompt, system_prompt)

    def _call_groq_api(self, prompt: str, system_prompt: str = None) -> str:
        """
        Call Groq API với retry logic và rate limiting để tránh lỗi 429
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            
        Returns:
            Generated text from LLM
        """
        if not self.api_key:
            return "Error: API key not configured. Please set GROQ_API_KEY."
        
        # Rate limiting: đợi nếu request quá nhanh
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            wait_time = self.request_delay - elapsed
            print(f"⏳ Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            time.sleep(wait_time)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # GPT-OSS works best with all instructions in user message
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        messages = [{"role": "user", "content": full_prompt}]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.6,
            "max_completion_tokens": 2048,
            "top_p": 0.95,
            "include_reasoning": True,
            "stream": False
        }
        
        # Retry logic với exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                # Update last request time
                self.last_request_time = time.time()
                
                # Check for rate limit error
                if response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff with jitter
                        wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                        print(f"⏳ Rate limit hit (429). Retrying in {wait_time:.1f}s... (Attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "Error: Rate limit exceeded. Please wait a moment and try again."
                
                response.raise_for_status()
                
                result = response.json()
                message = result["choices"][0]["message"]
                content = message.get("content", "")
                
                return content
                
            except requests.exceptions.HTTPError as e:
                if attempt < self.max_retries - 1 and e.response.status_code == 429:
                    wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                    print(f"⏳ HTTP 429 Error. Retrying in {wait_time:.1f}s... (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Error calling Groq API: {str(e)}"
            except requests.exceptions.RequestException as e:
                return f"Error calling Groq API: {str(e)}"
            except (KeyError, IndexError) as e:
                return f"Error parsing API response: {str(e)}"
        
        return "Error: Maximum retries exceeded."
    
    def interpret_shap_importance(self, shap_df: pd.DataFrame, model_metrics: Dict) -> str:
        """
        Giải thích SHAP feature importance với medical context chi tiết
        
        Args:
            shap_df: DataFrame với columns ['Feature', 'SHAP Importance']
            model_metrics: Dictionary chứa accuracy, precision, recall, f1
            
        Returns:
            Interpretation text
        """
        system_prompt = """You are a Cardiologist and Data Scientist expert, 
        specialized in explaining XAI (Explainable AI) analysis results for heart disease prediction.
        Explain EXTREMELY CLEARLY, EASILY UNDERSTANDABLE, with high medical professionalism."""
        
        top_features = shap_df.head(10).to_dict('records')
        
        # Add medical context for each feature
        features_with_context = []
        for feat in top_features:
            feat_name = feat['Feature']
            context = self.FEATURE_MEDICAL_CONTEXT.get(feat_name, 'No medical information available')
            features_with_context.append({
                'Feature': feat_name,
                'SHAP Importance': feat['SHAP Importance'],
                'Medical Context': context
            })
        
        prompt = f"""
Based on SHAP values analysis for heart disease prediction model:

**Model Metrics:**
- Accuracy: {model_metrics.get('Accuracy', 'N/A'):.4f}
- Precision: {model_metrics.get('Precision', 'N/A'):.4f}
- Recall: {model_metrics.get('Recall', 'N/A'):.4f}
- F1 Score: {model_metrics.get('F1 Score', 'N/A'):.4f}

**Top 10 most important features (by SHAP) with medical significance:**
{json.dumps(features_with_context, indent=2, ensure_ascii=False)}

Write a COMPLETE analysis report with the following structure (must follow strictly):

##  1. Overview of Most Important Indicators
- List top 5 features by importance order
- Briefly explain clinical significance of each indicator

## 🔬 2. In-Depth Analysis of Top Features
For EACH feature in top 5, analyze:
- **Why is it important?** (physiological/pathological mechanism)
- **Relationship with heart disease** (e.g., ST slope directly related to myocardial ischemia)
- **Danger thresholds** (if known - e.g., oldpeak >2mm is very dangerous)

## 💡 3. Medical Insights Derived
- Which feature groups often appear together? (e.g., high oldpeak + abnormal ST slope)
- Findings about "silent ischemia" (if chest pain type_4 appears)
- Role of ECG stress test in diagnosis

## 🏥 4. SPECIFIC Clinical Recommendations
Create recommendation table:

| Recommendation | Reason | Specific Action |
|----------------|--------|-----------------|
| (At least 5 specific recommendations for doctors/researchers) | | |

**Example recommendations:**
- Enhance ECG stress test → ST slope/oldpeak important → Perform exercise ECG for patients >45 years old
- Don't ignore asymptomatic patients → chest pain type_4 important → Add troponin test

## ⚕️ 5. Summary and Practical Implications
- Key points to remember
- Application in clinical examination workflow

**Requirements:**
- Write in CLEAR ENGLISH with medical terminology
- Use bullet points, tables, emojis for readability
- Explain ALL medical terms
- Provide SPECIFIC NUMBERS when possible (e.g., "age >45", "oldpeak >2mm")
"""
        
        return self._call_groq_api(prompt, system_prompt)
    
    def interpret_lime_instance(self, lime_result: Dict, feature_names: List[str]) -> str:
        """
        Giải thích LIME explanation cho một instance cụ thể
        
        Args:
            lime_result: Dictionary containing explanation_df, prediction, actual, etc.
            feature_names: List of feature names
            
        Returns:
            Interpretation text
        """
        system_prompt = """You are an expert in explaining ML model predictions for medical problems.
        Explain in a way that doctors or patients can understand."""
        
        exp_df = lime_result['explanation_df']
        prediction = lime_result['prediction']
        prediction_proba = lime_result['prediction_proba']
        actual = lime_result.get('actual', 'Unknown')
        is_correct = lime_result.get('correct', None)
        
        prompt = f"""
Explain the model's prediction for a specific patient:

**Prediction Information:**
- Model prediction: {"Heart Disease" if prediction == 1 else "No Heart Disease"}
- Confidence: {prediction_proba[1]*100:.1f}% (disease), {prediction_proba[0]*100:.1f}% (no disease)
- Actual result: {"Heart Disease" if actual == 1 else "No Heart Disease" if actual == 0 else "Unknown"}
- Prediction: {"Correct ✓" if is_correct else "Incorrect ✗" if is_correct is not None else "N/A"}

**Factors influencing this prediction (LIME analysis):**
{exp_df.to_string(index=False)}

Please explain:
1. Why did the model make this prediction?
2. Which factors push the prediction toward "disease" or "no disease"?
3. If the prediction is wrong, what could be the reason?
4. Advice for the patient based on this analysis

Answer in English, easy to understand, with practical value.
"""
        
        return self._call_groq_api(prompt, system_prompt)
    
    def interpret_permutation_importance(self, perm_df: pd.DataFrame) -> str:
        """
        Explain Permutation Importance
        
        Args:
            perm_df: DataFrame with columns ['Feature', 'Importance', 'Std']
            
        Returns:
            Interpretation text
        """
        system_prompt = """You are an ML expert explaining feature importance 
        in heart disease prediction problems."""
        
        top_features = perm_df.head(10).to_dict('records')
        
        prompt = f"""
Permutation Importance results for heart disease prediction model:

**Top 10 most important features:**
{json.dumps(top_features, indent=2, ensure_ascii=False)}

Please explain:
1. What's the difference between Permutation Importance and SHAP values?
2. Meaning of features with high importance
3. Which features can be removed to simplify the model?
4. Recommendations for future data collection

Answer in English, with clear structure.
"""
        
        return self._call_groq_api(prompt, system_prompt)
    
    def compare_methods(self, shap_df: pd.DataFrame, perm_df: pd.DataFrame) -> str:
        """
        So sánh kết quả giữa SHAP và Permutation Importance
        
        Args:
            shap_df: SHAP importance DataFrame
            perm_df: Permutation importance DataFrame
            
        Returns:
            Comparison interpretation
        """
        system_prompt = """You are an XAI expert, explaining differences between 
        explainability methods for ML models."""
        
        shap_top5 = shap_df.head(5)['Feature'].tolist()
        perm_top5 = perm_df.head(5)['Feature'].tolist()
        
        prompt = f"""
Compare feature importance results from two methods:

**SHAP Top 5 Features:**
{json.dumps(shap_top5, indent=2, ensure_ascii=False)}

**Permutation Importance Top 5 Features:**
{json.dumps(perm_top5, indent=2, ensure_ascii=False)}

Please analyze:
1. Agreement and differences between the two methods
2. Why are there differences? (if any)
3. Which method should be trusted more and in what cases?
4. Final conclusion about the most important features

Answer in English, with deep analysis.
"""
        
        return self._call_groq_api(prompt, system_prompt)
    
    def generate_full_report(self, xai_results: Dict, model_info: Dict) -> Dict[str, str]:
        """
        Generate complete XAI report with all interpretations
        
        Args:
            xai_results: Dictionary from XAIExplainer.full_xai_pipeline()
            model_info: Dictionary containing model name and metrics
            
        Returns:
            Dictionary containing all interpretations
        """
        print("\n" + "=" * 60)
        print("Generating LLM Interpretations...")
        print("=" * 60)
        
        report = {}
        
        # 1. SHAP Interpretation
        print("\n[1/4] Interpreting SHAP importance...")
        report['shap_interpretation'] = self.interpret_shap_importance(
            xai_results['shap_importance'],
            model_info
        )
        
        # 2. LIME Interpretations (first 2 instances)
        print("[2/4] Interpreting LIME explanations...")
        lime_interpretations = []
        for i, lime_result in enumerate(xai_results['lime_explanations'][:2]):
            interp = self.interpret_lime_instance(
                lime_result,
                xai_results['shap_importance']['Feature'].tolist()
            )
            lime_interpretations.append({
                'instance_idx': lime_result['instance_idx'],
                'interpretation': interp
            })
        report['lime_interpretations'] = lime_interpretations
        
        # 3. Permutation Importance
        print("[3/4] Interpreting permutation importance...")
        report['permutation_interpretation'] = self.interpret_permutation_importance(
            xai_results['permutation_importance']
        )
        
        # 4. Comparison
        print("[4/4] Comparing methods...")
        report['comparison'] = self.compare_methods(
            xai_results['shap_importance'],
            xai_results['permutation_importance']
        )
        
        print("\n" + "=" * 60)
        print("LLM Interpretation Completed!")
        print("=" * 60)
        
        return report


if __name__ == "__main__":
    # Test LLM interpreter
    import pandas as pd
    
    # Mock data for testing
    shap_df = pd.DataFrame({
        'Feature': ['max heart rate', 'oldpeak', 'age', 'ST slope_2'],
        'SHAP Importance': [0.15, 0.12, 0.10, 0.08]
    })
    
    model_metrics = {
        'Accuracy': 0.8765,
        'Precision': 0.8654,
        'Recall': 0.8821,
        'F1 Score': 0.8737
    }
    
    interpreter = LLMInterpreter()
    
    if interpreter.api_key:
        interpretation = interpreter.interpret_shap_importance(shap_df, model_metrics)
        print("\nSHAP Interpretation:")
        print(interpretation)
    else:
        print("\nPlease set GROQ_API_KEY environment variable to test LLM interpretation.")
        print("Get your free API key at: https://console.groq.com/")
