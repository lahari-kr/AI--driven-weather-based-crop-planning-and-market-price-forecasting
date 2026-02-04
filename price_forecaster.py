import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

class PriceForecaster:
    def __init__(self):
        self.models = {
            'arima': None,
            'lstm': None
        }
        self.scalers = {}
        self.metrics = {}
        self.history = None
    
    def calculate_accuracy_metrics(self, y_true, y_pred):
        """Calculate comprehensive accuracy metrics"""
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Accuracy percentage (100 - MAPE)
        accuracy = 100 - mape
        
        # Additional metrics
        # Mean Percentage Error (MPE) - shows bias
        mpe = np.mean((y_true - y_pred) / y_true) * 100
        
        # Symmetric MAPE (handles zero values better)
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'accuracy': float(accuracy),
            'mpe': float(mpe),
            'smape': float(smape)
        }
        
    def prepare_price_data(self, df, crop_name='Rice'):
        """Prepare price data for training"""
        # Filter by crop
        data = df[df['crop'] == crop_name].copy()
        data = data.sort_values('date')
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
        
        return data['price']
    
    def train_arima(self, price_series, order=(5,1,2)):
        """Train ARIMA model"""
        print("Training ARIMA model...")
        
        try:
            # Split data
            train_size = int(len(price_series) * 0.8)
            train, test = price_series[:train_size], price_series[train_size:]
            
            # Fit ARIMA
            model = ARIMA(train, order=order)
            fitted_model = model.fit()
            self.models['arima'] = fitted_model
            
            # Forecast
            forecast = fitted_model.forecast(steps=len(test))
            
            # Calculate comprehensive metrics
            metrics = self.calculate_accuracy_metrics(test.values, forecast.values)
            
            # Add ARIMA-specific metrics
            metrics['order'] = order
            metrics['aic'] = float(fitted_model.aic)
            metrics['bic'] = float(fitted_model.bic)
            
            self.metrics['arima'] = metrics
            
            print(f"ARIMA Performance:")
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  Accuracy: {metrics['accuracy']:.2f}%")
            
            return test, forecast
            
        except Exception as e:
            print(f"ARIMA training error: {e}")
            self.metrics['arima'] = {'error': str(e)}
            return None, None
    
    def prepare_lstm_data(self, price_series, lookback=30):
        """Prepare data for LSTM"""
        from sklearn.preprocessing import MinMaxScaler
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(price_series.values.reshape(-1, 1))
        self.scalers['lstm'] = scaler
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def train_lstm(self, price_series, lookback=30, epochs=50):
        """Train LSTM model"""
        print("Training LSTM model...")
        
        try:
            # Prepare data
            X, y = self.prepare_lstm_data(price_series, lookback)
            
            # Split
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])  
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Early stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            # Train
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=0
            )            
            self.models['lstm'] = model
            self.history = history
            
            # Predict
            predictions = model.predict(X_test, verbose=0)
            
            # Inverse transform
            scaler = self.scalers['lstm']
            predictions = scaler.inverse_transform(predictions)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate comprehensive metrics
            metrics = self.calculate_accuracy_metrics(
                y_test_actual.flatten(), 
                predictions.flatten()
            )
            
            # Add LSTM-specific metrics
            metrics['lookback'] = lookback
            metrics['epochs'] = len(history.history['loss'])
            metrics['final_loss'] = float(history.history['loss'][-1])
            
            self.metrics['lstm'] = metrics
            
            print(f"LSTM Performance:")
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  Accuracy: {metrics['accuracy']:.2f}%")
            
            return y_test_actual, predictions
            
        except Exception as e:
            print(f"LSTM training error: {e}")
            self.metrics['lstm'] = {'error': str(e)}
            return None, None
    
    def forecast_arima(self, steps=30):
        """Forecast using ARIMA"""
        if self.models['arima'] is None:
            raise ValueError("ARIMA model not trained")
        
        forecast = self.models['arima'].forecast(steps=steps)
        return forecast
    
    def forecast_lstm(self, price_series, steps=30, lookback=30):
        """Forecast using LSTM"""
        if self.models['lstm'] is None:
            raise ValueError("LSTM model not trained")
        
        model = self.models['lstm']
        scaler = self.scalers['lstm']
        
        # Get last lookback values
        scaled_data = scaler.transform(price_series.values[-lookback:].reshape(-1, 1))
        
        predictions = []
        current_batch = scaled_data.reshape((1, lookback, 1))
        
        for i in range(steps):
            # Predict next value
            pred = model.predict(current_batch, verbose=0)[0]
            predictions.append(pred)
            
            # Update batch
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
        
        # Inverse transform
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        return predictions.flatten()
    
    def get_comparison_metrics(self):
        """Get comparison metrics"""
        return self.metrics
    
    def print_model_comparison(self):
        """Print a formatted comparison of model performances"""
        print("\n" + "="*70)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*70)
        
        for model_name, metrics in self.metrics.items():
            if 'error' in metrics:
                print(f"\n{model_name.upper()}: Training failed - {metrics['error']}")
                continue
                
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:        {metrics.get('accuracy', 'N/A'):.2f}%")
            print(f"  MAPE:            {metrics.get('mape', 'N/A'):.2f}%")
            print(f"  MAE:             {metrics.get('mae', 'N/A'):.2f}")
            print(f"  RMSE:            {metrics.get('rmse', 'N/A'):.2f}")
            print(f"  R² Score:        {metrics.get('r2', 'N/A'):.4f}")
            print(f"  SMAPE:           {metrics.get('smape', 'N/A'):.2f}%")
        
        print("\n" + "="*70)
        
        # Determine best model
        if len(self.metrics) > 1:
            valid_models = {k: v for k, v in self.metrics.items() if 'error' not in v}
            if valid_models:
                best_model = max(valid_models.items(), 
                               key=lambda x: x[1].get('accuracy', 0))
                print(f"Best Model: {best_model[0].upper()} "
                      f"(Accuracy: {best_model[1]['accuracy']:.2f}%)")
                print("="*70 + "\n")
    
    def save_models(self, path='data/trained_models/price_models'):
        """Save models"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save ARIMA
        if self.models['arima']:
            joblib.dump(self.models['arima'], f'{path}/arima_model.pkl')
        
        # Save LSTM
        if self.models['lstm']:
            self.models['lstm'].save(f'{path}/lstm_model.h5')
            joblib.dump(self.scalers['lstm'], f'{path}/lstm_scaler.pkl')
        
        # Save metrics
        joblib.dump(self.metrics, f'{path}/metrics.pkl')
        
        print(f"Price models saved to {path}")
    
    def load_models(self, path='data/trained_models/price_models'):
        """Load models"""
        path_obj = Path(path)
        
        if not path_obj.exists():
            return False
        
        # Load ARIMA
        arima_path = path_obj / 'arima_model.pkl'
        if arima_path.exists():
            self.models['arima'] = joblib.load(arima_path)
        
        # Load LSTM
        lstm_path = path_obj / 'lstm_model.h5'
        scaler_path = path_obj / 'lstm_scaler.pkl'
        if lstm_path.exists() and scaler_path.exists():
            self.models['lstm'] = keras.models.load_model(lstm_path)
            self.scalers['lstm'] = joblib.load(scaler_path)
        
        # Load metrics
        metrics_path = path_obj / 'metrics.pkl'
        if metrics_path.exists():
            self.metrics = joblib.load(metrics_path)
        
        print(f"Price models loaded from {path}")
        return True