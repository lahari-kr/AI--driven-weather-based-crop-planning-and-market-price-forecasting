from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

# Import custom modules
from utils.weather import WeatherService
from utils.preprocessing import DataPreprocessor
from models.crop_predictor import CropPredictor
from models.price_forecaster import PriceForecaster

load_dotenv()

app = FastAPI(title="AgriAI API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
weather_service = WeatherService()
preprocessor = DataPreprocessor()
crop_predictor = CropPredictor()
price_forecaster = PriceForecaster()

# Global state
models_initialized = False

# Pydantic models
class LocationRequest(BaseModel):
    location: str

class CropPredictionRequest(BaseModel):
    soil_type: str
    temp_min: float
    temp_max: float
    humidity_min: float
    humidity_max: float
    model_type: str = 'random_forest'

class PriceForecastRequest(BaseModel):
    crop: str
    days: int = 30
    model_type: str = 'arima'

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global models_initialized
    
    print("Initializing models...")
    
    # Check if models exist, otherwise train them
    crop_model_path = 'data/trained_models/crop_models.pkl'
    price_model_path = 'data/trained_models/price_models'
    
    if not Path(crop_model_path).exists():
        print("Training crop prediction models...")
        # Load dataset
        df = pd.read_csv('data/SoilCrops_2000.csv')
        X, y = crop_predictor.prepare_data(df)
        crop_predictor.train_models(X, y)
        crop_predictor.save_models()
    else:
        print("Loading existing crop models...")
        crop_predictor.load_models()
    
    if not Path(price_model_path).exists():
        print("Training price forecasting models...")
        # Generate synthetic price data
        price_df = preprocessor.create_price_dataset(days=365*2, crop_name='Rice')
        price_series = price_forecaster.prepare_price_data(price_df)
        
        # Train both models
        price_forecaster.train_arima(price_series)
        price_forecaster.train_lstm(price_series)
        price_forecaster.save_models()
    else:
        print("Loading existing price models...")
        price_forecaster.load_models()
    
    models_initialized = True
    print("Models initialized successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AgriAI API",
        "version": "1.0.0",
        "endpoints": {
            "weather": "/api/weather",
            "location": "/api/location",
            "crop_predict": "/api/crop/predict",
            "price_forecast": "/api/price/forecast",
            "model_metrics": "/api/models/metrics"
        }
    }

@app.get("/api/weather")
async def get_weather(lat: float = Query(...), lon: float = Query(...)):
    """Get weather data for coordinates"""
    try:
        current = weather_service.get_current_weather(lat, lon)
        forecast = weather_service.get_forecast(lat, lon, days=7)
        
        return {
            "current": current,
            "forecast": forecast,
            "coordinates": {"lat": lat, "lon": lon}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/location")
async def get_location(request: LocationRequest):
    """Get coordinates from location name"""
    try:
        coords = weather_service.get_coordinates_from_location(request.location)
        
        if not coords:
            raise HTTPException(status_code=404, detail="Location not found")
        
        # Get weather for location
        weather = weather_service.get_current_weather(coords['lat'], coords['lon'])
        
        return {
            "coordinates": coords,
            "weather": weather
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/crop/predict")
async def predict_crop(request: CropPredictionRequest):
    """Predict suitable crops"""
    if not models_initialized:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    try:
        predictions = crop_predictor.predict_crops(
            soil_type=request.soil_type,
            temp_min=request.temp_min,
            temp_max=request.temp_max,
            humidity_min=request.humidity_min,
            humidity_max=request.humidity_max,
            model_type=request.model_type
        )
        
        return {
            "predictions": predictions,
            "model_used": request.model_type,
            "input": request.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/price/forecast")
async def forecast_price(request: PriceForecastRequest):
    """Forecast crop prices"""
    if not models_initialized:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    try:
        # Generate price data for the crop
        price_df = preprocessor.create_price_dataset(days=365, crop_name=request.crop)
        price_series = price_forecaster.prepare_price_data(price_df, request.crop)
        
        # Forecast
        if request.model_type == 'arima':
            predictions = price_forecaster.forecast_arima(steps=request.days)
        else:  # lstm
            predictions = price_forecaster.forecast_lstm(price_series, steps=request.days)
        
        # Create dates
        dates = pd.date_range(start=pd.Timestamp.now(), periods=request.days, freq='D')
        
        forecast_data = [
            {"date": str(date.date()), "price": float(price)}
            for date, price in zip(dates, predictions)
        ]
        
        return {
            "crop": request.crop,
            "model_used": request.model_type,
            "forecast": forecast_data,
            "summary": {
                "min_price": float(min(predictions)),
                "max_price": float(max(predictions)),
                "avg_price": float(sum(predictions) / len(predictions)),
                "current_price": float(price_series.iloc[-1])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/metrics")
async def get_model_metrics():
    """Get model performance metrics"""
    if not models_initialized:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    return {
        "crop_models": crop_predictor.get_comparison_metrics(),
        "price_models": price_forecaster.get_comparison_metrics()
    }

@app.get("/api/crops/list")
async def list_crops():
    """Get list of available crops"""
    crops = ['Rice', 'Wheat', 'Cotton', 'Maize', 'Sugarcane', 'Soybean', 
             'Groundnut', 'Potato', 'Tomato', 'Chickpea', 'Barley', 'Gram',
             'Jowar', 'Millet', 'Mustard', 'Pulses', 'Sunflower']
    return {"crops": crops}

@app.get("/api/soils/list")
async def list_soils():
    """Get list of soil types"""
    soils = ['Loamy', 'Clayey', 'Sandy', 'Black', 'Alluvial', 'Red', 'Clay']
    return {"soils": soils}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)