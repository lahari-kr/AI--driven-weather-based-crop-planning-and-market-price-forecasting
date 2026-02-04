import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ModelComparison:
    @staticmethod
    def compare_crop_models(metrics):
        """Compare crop prediction models"""
        comparison_data = []
        
        for model_name, model_metrics in metrics.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': model_metrics.get('accuracy', 0) * 100,
                'CV Mean': model_metrics.get('cv_mean', 0) * 100,
                'CV Std': model_metrics.get('cv_std', 0) * 100
            })
        
        df = pd.DataFrame(comparison_data)
        return df
    
    @staticmethod
    def plot_crop_model_comparison(metrics):
        """Create comparison plot for crop models"""
        df = ModelComparison.compare_crop_models(metrics)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Accuracy',
            x=df['Model'],
            y=df['Accuracy'],
            marker_color='lightgreen'
        ))
        
        fig.add_trace(go.Bar(
            name='Cross-Validation Mean',
            x=df['Model'],
            y=df['CV Mean'],
            marker_color='darkgreen'
        ))
        
        fig.update_layout(
            title='Crop Prediction Model Comparison',
            xaxis_title='Model',
            yaxis_title='Score (%)',
            barmode='group',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_feature_importance(metrics, model_name='random_forest'):
        """Plot feature importance"""
        if model_name not in metrics:
            return None
        
        importance = metrics[model_name].get('feature_importance', {})
        
        if not importance:
            return None
        
        df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        }).sort_values('Importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=df['Importance'],
            y=df['Feature'],
            orientation='h',
            marker_color='green'
        ))
        
        fig.update_layout(
            title=f'Feature Importance - {model_name.replace("_", " ").title()}',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=300
        )
        
        return fig
    
    @staticmethod
    def compare_price_models(metrics):
        """Compare price forecasting models"""
        comparison_data = []
        
        for model_name, model_metrics in metrics.items():
            if 'error' not in model_metrics:
                comparison_data.append({
                    'Model': model_name.upper(),
                    'MAE': model_metrics.get('mae', 0),
                    'RMSE': model_metrics.get('rmse', 0),
                    'R²': model_metrics.get('r2', 0)
                })
        
        df = pd.DataFrame(comparison_data)
        return df
    
    @staticmethod
    def plot_price_model_comparison(metrics):
        """Create comparison plot for price models"""
        df = ModelComparison.compare_price_models(metrics)
        
        if df.empty:
            return None
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Mean Absolute Error', 'Root Mean Squared Error', 'R² Score')
        )
        
        colors = ['blue', 'red']
        
        # MAE
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['MAE'], marker_color=colors, name='MAE'),
            row=1, col=1
        )
        
        # RMSE
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['RMSE'], marker_color=colors, name='RMSE'),
            row=1, col=2
        )
        
        # R²
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['R²'], marker_color=colors, name='R²'),
            row=1, col=3
        )
        
        fig.update_layout(
            title_text='Price Forecasting Model Comparison',
            showlegend=False,
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_forecast_comparison(actual, arima_pred, lstm_pred, dates=None):
        """Plot forecast comparison"""
        if dates is None:
            dates = list(range(len(actual)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        if arima_pred is not None and len(arima_pred) > 0:
            fig.add_trace(go.Scatter(
                x=dates[:len(arima_pred)],
                y=arima_pred,
                mode='lines+markers',
                name='ARIMA',
                line=dict(color='blue', width=2, dash='dash')
            ))
        
        if lstm_pred is not None and len(lstm_pred) > 0:
            fig.add_trace(go.Scatter(
                x=dates[:len(lstm_pred)],
                y=lstm_pred,
                mode='lines+markers',
                name='LSTM',
                line=dict(color='red', width=2, dash='dot')
            ))
        
        fig.update_layout(
            title='Price Forecast Comparison',
            xaxis_title='Time',
            yaxis_title='Price (₹/quintal)',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_metrics_table(crop_metrics, price_metrics):
        """Create comprehensive metrics table"""
        # Crop models
        crop_data = []
        for model, metrics in crop_metrics.items():
            crop_data.append({
                'Category': 'Crop Prediction',
                'Model': model.replace('_', ' ').title(),
                'Accuracy/R²': f"{metrics.get('accuracy', 0)*100:.2f}%",
                'CV Score': f"{metrics.get('cv_mean', 0)*100:.2f}%",
                'Additional': f"±{metrics.get('cv_std', 0)*100:.2f}%"
            })
        
        # Price models
        price_data = []
        for model, metrics in price_metrics.items():
            if 'error' not in metrics:
                price_data.append({
                    'Category': 'Price Forecasting',
                    'Model': model.upper(),
                    'Accuracy/R²': f"{metrics.get('r2', 0):.4f}",
                    'CV Score': f"MAE: {metrics.get('mae', 0):.2f}",
                    'Additional': f"RMSE: {metrics.get('rmse', 0):.2f}"
                })
        
        df = pd.DataFrame(crop_data + price_data)
        return df