"""
Visualization helpers cho XAI plots
Tái sử dụng trong Streamlit và Jupyter notebooks
"""

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
import numpy as np


class XAIVisualizer:
    """Helper class cho XAI visualizations"""
    
    def __init__(self, theme='plotly'):
        self.theme = theme
        
    def plot_shap_importance(self, shap_importance_df, top_n=10, title=None):
        """
        Plot SHAP feature importance
        
        Args:
            shap_importance_df: DataFrame với columns ['Feature', 'SHAP Importance']
            top_n: Số lượng features hiển thị
            title: Tiêu đề chart
        """
        top_features = shap_importance_df.head(top_n)
        
        fig = px.bar(
            top_features,
            x='SHAP Importance',
            y='Feature',
            orientation='h',
            title=title or f'Top {top_n} Most Important Features (SHAP)',
            color='SHAP Importance',
            color_continuous_scale='reds',
            text='SHAP Importance'
        )
        
        fig.update_traces(texttemplate='%{text:.4f}', textposition='inside')
        fig.update_layout(
            height=400 + (top_n * 20),
            yaxis={'categoryorder':'total ascending'},
            showlegend=False
        )
        
        return fig
    
    def plot_shap_summary(self, shap_values, features, feature_names):
        """
        Create SHAP summary plot (beeswarm)
        
        Args:
            shap_values: SHAP values array
            features: Feature values array
            feature_names: List of feature names
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            features,
            feature_names=feature_names,
            show=False,
            plot_size=(10, 8)
        )
        plt.tight_layout()
        return fig
    
    def plot_lime_explanation(self, lime_exp_df, title=None):
        """
        Plot LIME explanation cho một instance
        
        Args:
            lime_exp_df: DataFrame với columns ['Feature', 'Weight']
            title: Tiêu đề chart
        """
        exp_df = lime_exp_df.copy()
        exp_df['Direction'] = exp_df['Weight'].apply(
            lambda x: 'Towards Disease' if x > 0 else 'Towards No Disease'
        )
        exp_df = exp_df.sort_values('Weight')
        
        fig = px.bar(
            exp_df,
            x='Weight',
            y='Feature',
            orientation='h',
            color='Direction',
            title=title or 'LIME Feature Contributions',
            color_discrete_map={
                'Towards Disease': '#E74C3C',
                'Towards No Disease': '#27AE60'
            },
            text='Weight'
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)
        fig.update_traces(texttemplate='%{text:.3f}', textposition='inside')
        fig.update_layout(height=400)
        
        return fig
    
    def plot_permutation_importance(self, perm_df, top_n=10, title=None):
        """
        Plot Permutation Importance với error bars
        
        Args:
            perm_df: DataFrame với columns ['Feature', 'Importance', 'Std']
            top_n: Số lượng features
            title: Tiêu đề
        """
        top_features = perm_df.head(top_n)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_features['Feature'],
            x=top_features['Importance'],
            error_x=dict(type='data', array=top_features['Std']),
            orientation='h',
            marker=dict(
                color=top_features['Importance'],
                colorscale='blues',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=top_features['Importance'],
            texttemplate='%{text:.4f}',
            textposition='inside'
        ))
        
        fig.update_layout(
            title=title or f'Top {top_n} Features by Permutation Importance',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=400 + (top_n * 20),
            yaxis={'categoryorder':'total ascending'}
        )
        
        return fig
    
    def plot_model_comparison(self, results_df, metric='Accuracy', title=None):
        """
        So sánh performance của các models
        
        Args:
            results_df: DataFrame với columns ['Model Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
            metric: Metric để plot
            title: Tiêu đề
        """
        sorted_df = results_df.sort_values(metric, ascending=True)
        
        fig = px.bar(
            sorted_df,
            x=metric,
            y='Model Name',
            orientation='h',
            color=metric,
            text=metric,
            title=title or f'{metric} Comparison Across Models',
            color_continuous_scale='viridis'
        )
        
        fig.update_traces(texttemplate='%{text:.4f}', textposition='inside')
        fig.update_layout(height=400)
        
        return fig
    
    def plot_all_metrics(self, results_df, title=None):
        """
        Plot tất cả metrics cho tất cả models
        
        Args:
            results_df: DataFrame với model metrics
            title: Tiêu đề
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df['Model Name'],
                y=results_df[metric],
                text=results_df[metric].round(4),
                textposition='auto'
            ))
        
        fig.update_layout(
            title=title or 'Model Performance Comparison - All Metrics',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_confusion_matrix(self, conf_matrix, labels=['No Disease', 'Disease'], title=None):
        """
        Plot confusion matrix
        
        Args:
            conf_matrix: 2x2 confusion matrix
            labels: Class labels
            title: Tiêu đề
        """
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=conf_matrix,
            texttemplate='%{text}',
            textfont={"size": 20},
            showscale=True
        ))
        
        fig.update_layout(
            title=title or 'Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400,
            width=400
        )
        
        return fig
    
    def plot_feature_comparison(self, shap_df, perm_df, top_n=10):
        """
        So sánh SHAP và Permutation Importance side-by-side
        
        Args:
            shap_df: SHAP importance DataFrame
            perm_df: Permutation importance DataFrame
            top_n: Số features
        """
        # Get top features from both
        shap_top = shap_df.head(top_n)
        perm_top = perm_df.head(top_n)
        
        # Merge
        shap_top = shap_top.rename(columns={'SHAP Importance': 'SHAP'})
        perm_top = perm_top.rename(columns={'Importance': 'Permutation'})
        
        merged = pd.merge(
            shap_top[['Feature', 'SHAP']],
            perm_top[['Feature', 'Permutation']],
            on='Feature',
            how='outer'
        ).fillna(0)
        
        # Normalize for comparison
        merged['SHAP_norm'] = merged['SHAP'] / merged['SHAP'].max()
        merged['Perm_norm'] = merged['Permutation'] / merged['Permutation'].max()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='SHAP',
            y=merged['Feature'],
            x=merged['SHAP_norm'],
            orientation='h',
            marker_color='#E74C3C'
        ))
        
        fig.add_trace(go.Bar(
            name='Permutation',
            y=merged['Feature'],
            x=merged['Perm_norm'],
            orientation='h',
            marker_color='#3498DB'
        ))
        
        fig.update_layout(
            title='Feature Importance: SHAP vs Permutation (Normalized)',
            xaxis_title='Normalized Importance',
            yaxis_title='Feature',
            barmode='group',
            height=400 + (len(merged) * 20),
            yaxis={'categoryorder':'total ascending'}
        )
        
        return fig
    
    def plot_prediction_gauge(self, probability, title=None):
        """
        Gauge chart cho disease probability
        
        Args:
            probability: Xác suất có bệnh (0-1)
            title: Tiêu đề
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title or "Disease Probability (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=400)
        return fig


# Example usage
if __name__ == "__main__":
    # Create sample data
    shap_df = pd.DataFrame({
        'Feature': ['max heart rate', 'oldpeak', 'age', 'ST slope_2', 'exercise angina'],
        'SHAP Importance': [0.15, 0.12, 0.10, 0.08, 0.07]
    })
    
    visualizer = XAIVisualizer()
    fig = visualizer.plot_shap_importance(shap_df)
    fig.show()
