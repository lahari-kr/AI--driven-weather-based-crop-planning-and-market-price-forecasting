import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from pathlib import Path

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_soil_crop_data(self, filepath='data/SoilCrops_2000.csv'):
        """Load and preprocess soil-crop dataset"""
        df = pd.read_csv(filepath)
        return df
    
    def prepare_crop_features(self, df):
        """Prepare features for crop prediction"""
        # Encode categorical variables
        categorical_cols = ['Crop', 'SoilType']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def create_price_dataset(self, base_price=2000, days=365, crop_name='Rice'):
        """Generate synthetic historical price data with realistic patterns"""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        
        # Create seasonal pattern
        seasonal = 200 * np.sin(np.linspace(0, 4*np.pi, days))
        
        # Add trend
        trend = np.linspace(0, 500, days)
        
        # Add noise
        noise = np.random.normal(0, 100, days)
        
        # Combine components
        prices = base_price + seasonal + trend + noise
        prices = np.maximum(prices, base_price * 0.5)  # Floor price
        
        df = pd.DataFrame({
            'date': dates,
            'crop': crop_name,
            'price': prices,
            'volume': np.random.randint(1000, 5000, days)
        })
        
        return df
    
    def save_encoders(self, path='data/trained_models/encoders.pkl'):
        """Save label encoders"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.label_encoders, path)
    
    def load_encoders(self, path='data/trained_models/encoders.pkl'):
        """Load label encoders"""
        if Path(path).exists():
            self.label_encoders = joblib.load(path)
        return self.label_encoders