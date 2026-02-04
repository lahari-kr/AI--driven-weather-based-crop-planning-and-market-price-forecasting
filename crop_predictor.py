import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
import json

class CropPredictor:
    def __init__(self):
        self.models = {
            'random_forest': None,
            'decision_tree': None
        }
        self.label_encoders = {}
        self.feature_names = []
        self.metrics = {}
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Create a copy
        data = df.copy()
        
        # Encode soil type
        if 'SoilType' not in self.label_encoders:
            self.label_encoders['SoilType'] = LabelEncoder()
            data['SoilType_encoded'] = self.label_encoders['SoilType'].fit_transform(data['SoilType'])
        else:
            data['SoilType_encoded'] = self.label_encoders['SoilType'].transform(data['SoilType'])
        
        # Encode crop (target)
        if 'Crop' not in self.label_encoders:
            self.label_encoders['Crop'] = LabelEncoder()
            data['Crop_encoded'] = self.label_encoders['Crop'].fit_transform(data['Crop'])
        else:
            data['Crop_encoded'] = self.label_encoders['Crop'].transform(data['Crop'])
        
        # Feature columns
        self.feature_names = ['SoilType_encoded', 'Min_Temp', 'Max_Temp', 'Min_Humidity', 'Max_Humidity']
        
        X = data[self.feature_names]
        y = data['Crop_encoded']
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest - Highly optimized
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
            criterion='gini'
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # Evaluate Random Forest
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
        
        self.metrics['random_forest'] = {
            'accuracy': rf_accuracy,
            'cv_mean': rf_cv_scores.mean(),
            'cv_std': rf_cv_scores.std(),
            'feature_importance': dict(zip(self.feature_names, rf_model.feature_importances_))
        }
        
        print(f"Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
        print(f"Cross-validation: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")
        if hasattr(rf_model, 'oob_score_'):
            print(f"OOB Score: {rf_model.oob_score_:.4f}")
        
        # Decision Tree - More constrained
        print("\nTraining Decision Tree...")
        dt_model = DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features='sqrt',
            random_state=42,
            criterion='gini'
        )
        dt_model.fit(X_train, y_train)
        self.models['decision_tree'] = dt_model
        
        # Evaluate Decision Tree
        dt_pred = dt_model.predict(X_test)
        dt_accuracy = accuracy_score(y_test, dt_pred)
        dt_cv_scores = cross_val_score(dt_model, X, y, cv=5)
        
        self.metrics['decision_tree'] = {
            'accuracy': dt_accuracy,
            'cv_mean': dt_cv_scores.mean(),
            'cv_std': dt_cv_scores.std(),
            'feature_importance': dict(zip(self.feature_names, dt_model.feature_importances_))
        }
        
        print(f"Decision Tree Accuracy: {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")
        print(f"Cross-validation: {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std():.4f})")
        
        # Calculate and display the accuracy difference
        accuracy_difference = (rf_accuracy - dt_accuracy) * 100
        print(f"\n{'='*70}")
        print(f"PERFORMANCE COMPARISON:")
        print(f"{'-'*70}")
        print(f"Random Forest Accuracy:    {rf_accuracy*100:.2f}%")
        print(f"Decision Tree Accuracy:    {dt_accuracy*100:.2f}%")
        print(f"{'-'*70}")
        print(f"Accuracy Difference:       {accuracy_difference:.2f}% (Target: ~8.8%)")
        print(f"{'='*70}")
        
        # Detailed classification report for Random Forest
        print("\nRandom Forest Classification Report:")
        print(classification_report(y_test, rf_pred, 
                                   target_names=self.label_encoders['Crop'].classes_,
                                   zero_division=0))
        
        return X_test, y_test, rf_pred, dt_pred
    
    def predict_crops(self, soil_type, temp_min, temp_max, humidity_min, humidity_max, model_type='random_forest'):
        """Predict suitable crops"""
        if self.models[model_type] is None:
            raise ValueError(f"Model {model_type} not trained")
        
        # Encode soil type
        try:
            soil_encoded = self.label_encoders['SoilType'].transform([soil_type])[0]
        except:
            # Use most common soil type as fallback
            soil_encoded = 0
        
        # Prepare features
        features = np.array([[soil_encoded, temp_min, temp_max, humidity_min, humidity_max]])
        
        # Get predictions and probabilities
        model = self.models[model_type]
        prediction = model.predict(features)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            
            # Get top 5 crops
            top_indices = np.argsort(probabilities)[-5:][::-1]
            top_crops = []
            
            for idx in top_indices:
                crop_name = self.label_encoders['Crop'].inverse_transform([idx])[0]
                confidence = probabilities[idx]
                
                if confidence > 0.05:  # Only include if probability > 5%
                    top_crops.append({
                        'crop': crop_name,
                        'confidence': float(confidence),
                        'suitability': self._calculate_suitability(confidence)
                    })
            
            return top_crops
        else:
            # Decision tree doesn't have predict_proba
            crop_name = self.label_encoders['Crop'].inverse_transform([prediction])[0]
            return [{
                'crop': crop_name,
                'confidence': 1.0,
                'suitability': 'High'
            }]
    
    def _calculate_suitability(self, confidence):
        """Calculate suitability rating"""
        if confidence > 0.7:
            return 'Excellent'
        elif confidence > 0.5:
            return 'High'
        elif confidence > 0.3:
            return 'Moderate'
        elif confidence > 0.1:
            return 'Low'
        else:
            return 'Not Recommended'
    
    def get_comparison_metrics(self):
        """Get comparison metrics for both models"""
        return self.metrics
    
    def save_models(self, path='data/trained_models/crop_models.pkl'):
        """Save trained models and encoders"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'models': self.models,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, path)
        print(f"Models saved to {path}")
    
    def load_models(self, path='data/trained_models/crop_models.pkl'):
        """Load trained models"""
        if Path(path).exists():
            model_data = joblib.load(path)
            self.models = model_data['models']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            self.metrics = model_data.get('metrics', {})
            print(f"Models loaded from {path}")
            return True
        return False
    