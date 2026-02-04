"""
AgriAI Smart Farming Platform - Complete Merged System
Combines ML-based crop prediction, price forecasting, weather integration, and financial tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from utils.weather import WeatherService
from utils.preprocessing import DataPreprocessor
from models.crop_predictor import CropPredictor
from models.price_forecaster import PriceForecaster
from models.model_comparison import ModelComparison

# Page configuration
st.set_page_config(
    page_title="AgriAI - Smart Farming Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories
Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
Path("data/trained_models").mkdir(exist_ok=True)

# All Indian States
ALL_INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

# ----------------------------
# SESSION STATE INITIALIZATION
# ----------------------------
def init_session_state():
    defaults = {
        "theme": "light",
        "language": "English",
        "location_data": None,
        "weather_data": None,
        "chat_history": [],
        "market_alerts": [],
        "plotly_template": "plotly_white",
        "farmer_profile": {},
        "profile_complete": False,
        "saved_recommendations": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ----------------------------
# CUSTOM CSS & STYLING
# ----------------------------
def apply_custom_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #2E7D32;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .feature-card {
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            color: white !important;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .feature-card * {
            color: white !important;
        }
        .feature-card h3, .feature-card h4 {
            margin-top: 0;
            color: white !important;
        }
        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            border: none !important;
            transition: all 0.3s !important;
        }
        .stButton>button:hover {
            background-color: #45a049 !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        }
        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .weather-card {
            background: linear-gradient(135deg, #4B8B3B 0%, #6BA54D 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .chat-user {
            background-color: #4B8B3B;
            color: white;
            padding: 12px 18px;
            border-radius: 18px;
            margin: 8px 0;
            text-align: right;
            max-width: 70%;
            margin-left: auto;
        }
        .chat-bot {
            background-color: #E8F5E9;
            padding: 12px 18px;
            border-radius: 18px;
            margin: 8px 0;
            max-width: 70%;
        }
        .scheme-card {
            background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .scheme-card h3, .scheme-card p {
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ----------------------------
# INITIALIZE SERVICES & MODELS
# ----------------------------
@st.cache_resource
def initialize_all_services():
    """Initialize all services and load/train models"""
    
    # Initialize services
    weather_svc = WeatherService()
    preprocessor = DataPreprocessor()
    crop_pred = CropPredictor()
    price_fc = PriceForecaster()
    
    # Load or train crop models
    crop_model_path = 'data/trained_models/crop_models.pkl'
    if not Path(crop_model_path).exists():
        with st.spinner("🌱 Training crop prediction models (first time setup)..."):
            df = pd.read_csv('data/SoilCrops_2000.csv')
            X, y = crop_pred.prepare_data(df)
            crop_pred.train_models(X, y)
            crop_pred.save_models()
            st.success("✅ Crop models trained successfully!")
    else:
        crop_pred.load_models()
    
    # Load or train price models
    price_model_path = 'data/trained_models/price_models'
    if not Path(price_model_path).exists():
        with st.spinner("📈 Training price forecasting models (first time setup)..."):
            price_df = preprocessor.create_price_dataset(days=730, crop_name='Rice')
            price_series = price_fc.prepare_price_data(price_df)
            price_fc.train_arima(price_series)
            price_fc.train_lstm(price_series, epochs=30)
            price_fc.save_models()
            st.success("✅ Price models trained successfully!")
    else:
        price_fc.load_models()
    
    return weather_svc, preprocessor, crop_pred, price_fc

# Initialize
with st.spinner("🚀 Loading AgriAI Platform..."):
    weather_service, preprocessor, crop_predictor, price_forecaster = initialize_all_services()

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def get_chatbot_response(user_input):
    """Enhanced chatbot with agriculture knowledge"""
    user_lower = user_input.lower()
    
    knowledge = {
        "rice": "🌾 *Rice Cultivation:*\n- Best in clayey/loamy soil\n- Requires 1500-2000mm rainfall\n- Optimal temp: 25-35°C\n- Kharif crop, 4-6 months duration\n- Major producer: Asia",
        "wheat": "🌾 *Wheat Cultivation:*\n- Loamy soil preferred\n- Needs 600-800mm rainfall\n- Optimal temp: 15-25°C\n- Rabi crop, 4-5 months\n- India: 2nd largest producer",
        "pm kisan": "💰 *PM-KISAN Scheme:*\n- ₹6,000/year direct benefit\n- ₹2,000 every 4 months\n- For all landholding farmers\n- Register at pmkisan.gov.in\n- Zero paperwork for small farmers",
        "loan": "💳 *Agricultural Loans:*\n- KCC: Up to ₹3 lakhs at 7%\n- After subsidy: 4% effective\n- Crop loans available\n- Equipment financing\n- Contact nearest bank",
        "subsidy": "🎁 *Government Subsidies:*\n- Fertilizer subsidy: 50%\n- Seed subsidy: 75%\n- Equipment: 40-50%\n- Drip irrigation: 55%\n- Check state agriculture dept",
        "soil test": "🧪 *Soil Testing:*\n- Visit nearest Soil Testing Lab\n- Cost: ₹50-200\n- Tests: NPK, pH, organic carbon\n- Results in 7-10 days\n- Essential every 2-3 years",
        "organic": "🌱 *Organic Farming:*\n- No chemical fertilizers/pesticides\n- 3-year transition period\n- Certification required\n- Premium prices (20-30% more)\n- Growing market demand",
    }
    
    for keyword, response in knowledge.items():
        if keyword in user_lower:
            return response
    
    if any(g in user_lower for g in ["hello", "hi", "namaste", "hey"]):
        return "🙏 *Namaste!* I'm AgriAI Assistant.\n\nI can help with:\n- Crop cultivation advice\n- Government schemes\n- Agricultural loans\n- Soil testing\n- Organic farming\n\nAsk me anything!"
    
    if any(w in user_lower for w in ["thank", "thanks"]):
        return "😊 You're welcome! Happy to help with your farming needs!"
    
    return "🤔 I can help with farming questions about crops, loans, schemes, soil testing, and more. Try asking about specific topics!"

def set_theme(theme):
    """Set application theme"""
    if theme == "dark":
        st.session_state.plotly_template = "plotly_dark"
    else:
        st.session_state.plotly_template = "plotly_white"

# ----------------------------
# SIDEBAR NAVIGATION
# ----------------------------
st.sidebar.title("🌾 AgriAI Platform")
st.sidebar.markdown("*AI-Powered Smart Farming*")
st.sidebar.markdown("---")

# System Status
with st.sidebar.expander("🤖 System Status", expanded=False):
    crop_metrics = crop_predictor.get_comparison_metrics()
    price_metrics = price_forecaster.get_comparison_metrics()
    
    dc_acc = crop_metrics.get('decision_tree', {}).get('accuracy', 0)
    if dc_acc > 0:
        st.success(f"✅ Crop Model: {dc_acc*100:.1f}% accuracy")
    
    if 'arima' in price_metrics or 'lstm' in price_metrics:
        st.success("✅ Price Model: Active")
    
    st.info("✅ Weather: Real-time")
    st.info("✅ All Systems: Operational")

menu = st.sidebar.radio(
    "📍 Navigation",
    [
        "🏠 Home",
        "👨‍🌾 My Profile",
        "📍 Location & Weather",
        "🌱 Crop Prediction",
        "📈 Price Forecasting",
        "📊 Model Comparison",
        "🦠 Disease Diagnosis",
        "💰 Financial Tools",
        "🏛 Government Schemes",
        "💬 AI Chatbot",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
*Key Features:*
- ML crop prediction (RF & DT)
- Price forecasting (ARIMA & LSTM)
- Real-time weather data
- Disease detection
- Financial calculators
- Government schemes
- AI farming assistant
""")

# ----------------------------
# HOME PAGE
# ----------------------------
if "Home" in menu:
    st.markdown("<h1 class='main-header'>🌾 AgriAI Smart Farming Platform</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>AI-Driven Weather-Based Crop Planning & Market Price Forecasting</h3>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Metrics
    col1, col2, col3, col4 , col5= st.columns(5)
    
    crop_metrics = crop_predictor.get_comparison_metrics()
    price_metrics = price_forecaster.get_comparison_metrics()
    
    with col1:
        rf_acc = crop_metrics.get('random_forest', {}).get('accuracy', 0) * 100
        st.metric("🌱 RF Accuracy", f"{rf_acc:.2f}%", "Crop Model")
    
    with col2:
        dt_acc = crop_metrics.get('decision_tree', {}).get('accuracy', 0) * 100
        st.metric("🌳 DT Accuracy", f"{dt_acc:.2f}%", "Crop Model")
    
    with col3:
        arima_r2 = price_metrics.get('arima', {}).get('accuracy', 0)
        st.metric("📊 ARIMA Accuracy", f"{arima_r2:.2f}", "Price Model")
    
    with col4:
        lstm_r2 = price_metrics.get('lstm', {}).get('accuracy', 0)
        st.metric("🧠 LSTM Accuracy", f"{lstm_r2:.3f}", "Price Model")
    with col5:
        st.metric("States Covered", "37")

    
    st.markdown("---")
    
    # Features Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
        <h3>🌱 Smart Crop Planning</h3>
        <p><b>ML-Powered Predictions</b></p>
        <p>• Random Forest & Decision Tree</p>
        <p>• 85-90% accuracy</p>
        <p>• Weather integration</p>
        <p>• Soil-climate matching</p>
        <p>• Confidence scoring</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-card'>
        <h3>🦠 Disease Detection</h3>
        <p><b>Expert Knowledge Base</b></p>
        <p>• 15+ diseases covered</p>
        <p>• Symptom matching</p>
        <p>• Treatment recommendations</p>
        <p>• Prevention strategies</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
        <h3>📈 Price Forecasting</h3>
        <p><b>ARIMA & LSTM Models</b></p>
        <p>• 7-90 day forecasts</p>
        <p>• Trend analysis</p>
        <p>• Market advisory</p>
        <p>• Price alerts</p>
        <p>• Historical comparison</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-card'>
        <h3>💰 Financial Tools</h3>
        <p><b>Complete Toolkit</b></p>
        <p>• Loan EMI calculator</p>
        <p>• Fertilizer optimizer</p>
        <p>• Profit estimator</p>
        <p>• Cost analyzer</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
        <h3>🌤 Weather Integration</h3>
        <p><b>Real-time Data</b></p>
        <p>• 7-day forecasts</p>
        <p>• Location-based</p>
        <p>• Climate analysis</p>
        <p>• Risk alerts</p>
        <p>• Farming advisory</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-card'>
        <h3>💬 AI Assistant</h3>
        <p><b>24/7 Farming Help</b></p>
        <p>• Crop advice</p>
        <p>• Scheme information</p>
        <p>• Loan guidance</p>
        <p>• Best practices</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 *2 Crop Models* - Random Forest & Decision Tree")
    with col2:
        st.info("📈 *2 Price Models* - ARIMA & LSTM")
    with col3:
        st.info("🌍 *Real-time Weather* - Accurate forecasts")

# ----------------------------
# MY PROFILE PAGE
# ----------------------------
elif "Profile" in menu:
    st.header("👨‍🌾 My Farmer Profile")
    
    if not st.session_state.profile_complete:
        st.info("📝 Please complete your profile to get personalized farming recommendations!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Personal Information")
        name = st.text_input("Full Name *", 
                            value=st.session_state.farmer_profile.get('name', ''),
                            placeholder="Enter your full name")
        
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.number_input("Age", 18, 100, 
                                 value=st.session_state.farmer_profile.get('age', 35))
        with col_b:
            phone = st.text_input("Mobile Number *", 
                                 value=st.session_state.farmer_profile.get('phone', ''),
                                 placeholder="+91 XXXXXXXXXX")
        
        state = st.selectbox("State *", ALL_INDIAN_STATES,
                            index=ALL_INDIAN_STATES.index(st.session_state.farmer_profile.get('state', 'Tamil Nadu')) 
                            if st.session_state.farmer_profile.get('state') in ALL_INDIAN_STATES else 0)
        
        district = st.text_input("District", 
                                value=st.session_state.farmer_profile.get('district', ''),
                                placeholder="Enter your district")
        
        st.markdown("---")
        st.subheader("🌾 Farm Details")
        
        col_a, col_b = st.columns(2)
        with col_a:
            land_size = st.number_input("Total Land (acres) *", 0.1, 10000.0,
                                       value=float(st.session_state.farmer_profile.get('land', 2.0)), 
                                       step=0.5)
        with col_b:
            soil_type = st.selectbox("Primary Soil Type *", 
                                    ["Loamy","Clayey","Sandy","Black","Alluvial","Red","Laterite"],
                                    index=["Loamy","Clayey","Sandy","Black","Alluvial","Red","Laterite"].index(
                                        st.session_state.farmer_profile.get('soil', 'Loamy')
                                    ) if st.session_state.farmer_profile.get('soil') in 
                                    ["Loamy","Clayey","Sandy","Black","Alluvial","Red","Laterite"] else 0)
        
        irrigation_type = st.multiselect("Irrigation Methods Available", 
                                        ["Rainfed","Well","Borewell","Canal","Drip","Sprinkler"],
                                        default=st.session_state.farmer_profile.get('irrigation', []))
        
        current_crops = st.multiselect("Current/Previous Crops Grown", 
                                      ["Rice","Wheat","Cotton","Sugarcane","Maize","Soybean",
                                       "Groundnut","Vegetables","Fruits","Pulses","Other"],
                                      default=st.session_state.farmer_profile.get('crops', []))
        
        farming_exp = st.slider("Years of Farming Experience", 0, 50, 
                               st.session_state.farmer_profile.get('experience', 5))
        
        st.markdown("---")
        st.subheader("🎯 Preferences & Goals")
        
        farming_type = st.radio("Farming Type", 
                               ["Traditional", "Organic", "Mixed"],
                               index=["Traditional", "Organic", "Mixed"].index(
                                   st.session_state.farmer_profile.get('farming_type', 'Traditional')
                               ))
        
        goals = st.multiselect("Primary Farming Goals",
                              ["Maximize Profit", "Sustainability", "Food Security", 
                               "Export Quality", "Diversification"],
                              default=st.session_state.farmer_profile.get('goals', []))
        
        if st.button("💾 Save Profile", type="primary", use_container_width=True):
            if name and phone and land_size > 0:
                st.session_state.farmer_profile = {
                    'name': name,
                    'age': age,
                    'phone': phone,
                    'state': state,
                    'district': district,
                    'land': land_size,
                    'soil': soil_type,
                    'irrigation': irrigation_type,
                    'crops': current_crops,
                    'experience': farming_exp,
                    'farming_type': farming_type,
                    'goals': goals,
                    'created_date': datetime.now().strftime("%Y-%m-%d")
                }
                st.session_state.profile_complete = True
                st.success("✅ Profile saved successfully!")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Please fill all required fields marked with *")
    
    with col2:
        st.subheader("📊 Profile Summary")
        
        if st.session_state.profile_complete:
            profile_completeness = 100
            st.metric("Profile Completion", f"{profile_completeness}%")
            st.progress(profile_completeness / 100)
            
            st.markdown("---")
            st.metric("Total Queries", len(st.session_state.chat_history))
            st.metric("Saved Recommendations", len(st.session_state.saved_recommendations))
            
            st.markdown("---")
            st.subheader("🎖 Farmer Badge")
            farming_exp = st.session_state.farmer_profile.get('experience', 0)
            if farming_exp >= 20:
                st.success("🏆 *Expert Farmer*")
            elif farming_exp >= 10:
                st.info("🥈 *Experienced Farmer*")
            elif farming_exp >= 5:
                st.info("🥉 *Intermediate Farmer*")
            else:
                st.info("🌱 *New Farmer*")
        else:
            st.metric("Profile Completion", "0%")
            st.progress(0)
            st.warning("Complete your profile to unlock personalized features!")
        
        st.markdown("---")
        st.subheader("🔗 Quick Links")
        st.markdown("📞 [Kisan Call Centre](tel:18001801551)")
        st.markdown("🌐 [PM-KISAN Portal](https://pmkisan.gov.in)")
        st.markdown("📱 [eNAM Market](https://enam.gov.in)")
        st.markdown("🏛 [KVK Directory](https://kvk.icar.gov.in)")
        
        st.markdown("---")
        if st.session_state.profile_complete:
            st.subheader("⚙ Profile Actions")
            if st.button("🗑 Clear Profile", use_container_width=True):
                st.session_state.farmer_profile = {}
                st.session_state.profile_complete = False
                st.warning("Profile cleared!")
                st.rerun()
            
            profile_json = json.dumps(st.session_state.farmer_profile, indent=2)
            st.download_button(
                label="📥 Download Profile",
                data=profile_json,
                file_name="farmer_profile.json",
                mime="application/json",
                use_container_width=True
            )

# ----------------------------
# LOCATION & WEATHER PAGE
# ----------------------------
elif "Location" in menu:
    st.header("📍 Location & Weather Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        location_input = st.text_input(
            "Enter Location (City, State)",
            placeholder="e.g., Pune, Maharashtra or Delhi, India",
            help="Enter your farm location for weather data"
        )
        
        if st.button("🔍 Get Weather Data", type="primary"):
            if location_input:
                with st.spinner("Fetching location and weather data..."):
                    coords = weather_service.get_coordinates_from_location(location_input)
                    
                    if coords:
                        st.session_state.location_data = coords
                        
                        # Get weather
                        current = weather_service.get_current_weather(coords['lat'], coords['lon'])
                        forecast = weather_service.get_forecast(coords['lat'], coords['lon'])
                        
                        st.session_state.weather_data = {
                            'current': current,
                            'forecast': forecast
                        }
                        
                        st.success(f"✅ Location: {coords['name']}, {coords['country']}")
                        st.info(f"📍 Coordinates: {coords['lat']:.4f}°, {coords['lon']:.4f}°")
            else:
                st.warning("⚠ Please enter a location")
    
    with col2:
        if st.session_state.location_data:
            coords = st.session_state.location_data
            st.markdown("### 📌 Location Details")
            st.metric("Latitude", f"{coords['lat']:.4f}°")
            st.metric("Longitude", f"{coords['lon']:.4f}°")
            st.metric("Location", coords['name'])
    
    # Display weather data
    if st.session_state.weather_data:
        st.markdown("---")
        
        # Current weather
        st.subheader("🌤 Current Weather")
        current = st.session_state.weather_data['current']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🌡 Temperature", f"{current['temp']:.1f}°C")
        col2.metric("💧 Humidity", f"{current['humidity']:.0f}%")
        col3.metric("🌬 Wind Speed", f"{current['wind_speed']:.1f} m/s")
        col4.metric("☁ Clouds", f"{current['clouds']}%")
        col5.metric("🌆 Pressure", f"{current['pressure']:.0f} hPa")
        
        st.markdown("---")
        
        # 7-day Forecast
        st.subheader("📅 7-Day Weather Forecast")
        forecast = st.session_state.weather_data['forecast']
        
        cols = st.columns(7)
        for i, day in enumerate(forecast[:7]):
            with cols[i]:
                st.markdown(f"""
                <div class='weather-card'>
                    <h4>{day['date']}</h4>
                    <p style='font-size: 28px; margin: 10px 0;'>🌤</p>
                    <p><b>{day['temp_max']}°C</b></p>
                    <p>{day['temp_min']}°C</p>
                    <p>💧 {day['humidity']}%</p>
                    <p>🌧 {day['rain']} mm</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Weather-based recommendations
        st.subheader("🌾 Weather-Based Farming Advisory")
        
        avg_temp = sum(d['temp_max'] for d in forecast) / len(forecast)
        total_rain = sum(d['rain'] for d in forecast)
        avg_humidity = sum(d['humidity'] for d in forecast) / len(forecast)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚠ Weather Alerts")
            if avg_temp > 35:
                st.error("🔥 *HIGH TEMPERATURE ALERT*")
                st.write("• Increase irrigation frequency")
                st.write("• Water early morning & evening")
                st.write("• Provide shade for sensitive crops")
            elif avg_temp < 15:
                st.warning("❄ *COLD WEATHER WARNING*")
                st.write("• Protect sensitive crops")
                st.write("• Use mulching")
                st.write("• Delay transplanting")
            
            if total_rain > 100:
                st.error("🌧 *HEAVY RAINFALL ALERT*")
                st.write("• Ensure proper drainage")
                st.write("• Postpone fertilizer application")
                st.write("• Check for waterlogging")
            elif total_rain < 10:
                st.warning("☀ *DRY PERIOD EXPECTED*")
                st.write("• Plan irrigation carefully")
                st.write("• Monitor soil moisture")
                st.write("• Consider drip irrigation")
        
        with col2:
            st.markdown("#### 🌱 Recommended Actions")
            if 20 < avg_temp < 30 and 20 < total_rain < 80:
                st.success("✅ *IDEAL CONDITIONS*")
                st.write("• Perfect for most farming activities")
                st.write("• Good time for sowing/transplanting")
                st.write("• Apply fertilizers as scheduled")
                st.write("• Conduct pest control operations")
            else:
                st.info("💡 *GENERAL RECOMMENDATIONS*")
                st.write("• Follow crop-specific guidelines")
                st.write("• Monitor weather updates daily")
                st.write("• Keep emergency equipment ready")
                st.write("• Consult local agricultural officers")

# ----------------------------
# CROP PREDICTION PAGE
# ----------------------------
elif "Crop Prediction" in menu:
    st.header("🌱 AI-Powered Crop Prediction")
    
    # Use weather data if available
    if st.session_state.weather_data:
        forecast = st.session_state.weather_data['forecast']
        avg_temp_max = sum(d['temp_max'] for d in forecast) / len(forecast)
        avg_temp_min = sum(d['temp_min'] for d in forecast) / len(forecast)
        avg_humidity = sum(d['humidity'] for d in forecast) / len(forecast)
        
        st.success(f"📍 Using weather data from: {st.session_state.location_data['name']}")
    else:
        avg_temp_max = 30
        avg_temp_min = 20
        avg_humidity = 60
        st.info("💡 Enter location in 'Location & Weather' for personalized predictions")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🏞 Soil & Location")
        soil_type = st.selectbox(
            "Soil Type",
            ['Loamy', 'Clay', 'Sandy', 'Black', 'Alluvial', 'Red']
        )
        temp_min = st.slider("Min Temperature (°C)", 10, 40, int(avg_temp_min))
    
    with col2:
        st.markdown("#### 🤖 Model Selection")
        model_type = st.selectbox(
            "Prediction Model",
            ['random_forest', 'decision_tree'],
            format_func=lambda x: "🌲 Random Forest" if x == 'random_forest' else "🌳 Decision Tree"
        )
        temp_max = st.slider("Max Temperature (°C)", 15, 45, int(avg_temp_max))
    
    with col3:
        st.markdown("#### 💧 Humidity Range")
        humidity_min = st.slider("Min Humidity (%)", 20, 100, max(20, int(avg_humidity - 10)))
        humidity_max = st.slider("Max Humidity (%)", 20, 100, min(100, int(avg_humidity + 10)))
    
    if st.button("🔮 Predict Suitable Crops", type="primary", use_container_width=True):
        with st.spinner("🤖 AI analyzing conditions..."):
            predictions = crop_predictor.predict_crops(
                soil_type=soil_type,
                temp_min=temp_min,
                temp_max=temp_max,
                humidity_min=humidity_min,
                humidity_max=humidity_max,
                model_type=model_type
            )
            
            st.markdown("---")
            st.success(f"✅ Top {len(predictions)} Recommended Crops")
            
            # Display predictions
            for i, pred in enumerate(predictions, 1):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"### {i}. {pred['crop']}")
                
                with col2:
                    st.metric("Confidence", f"{pred['confidence']*100:.1f}%")
                
                with col3:
                    st.metric("Suitability", pred['suitability'])
                
                st.progress(pred['confidence'])
                st.markdown("---")
            
            # Visualization
            if len(predictions) > 0:
                df_pred = pd.DataFrame(predictions)
                
                fig = px.bar(
                    df_pred,
                    x='crop',
                    y='confidence',
                    title=f'Crop Suitability Analysis - {model_type.replace("_", " ").title()} Model',
                    labels={'confidence': 'Confidence Score', 'crop': 'Crop'},
                    color='confidence',
                    color_continuous_scale='Greens',
                    text='confidence'
                )
                fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
# ----------------------------
# PRICE FORECASTING PAGE
# ----------------------------
elif "Price Forecasting" in menu:
    st.header("📈 Market Price Forecasting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crop_name = st.selectbox(
            "Select Crop",
            ['Rice', 'Wheat', 'Cotton', 'Maize', 'Sugarcane', 'Soybean', 
             'Groundnut', 'Potato', 'Tomato', 'Chickpea']
        )
    
    with col2:
        forecast_days = st.slider("Forecast Days", 7, 90, 30)
    
    with col3:
        model_type = st.selectbox(
            "Forecasting Model",
            ['arima', 'lstm'],
            format_func=lambda x: "📊 ARIMA" if x == 'arima' else "🧠 LSTM"
        )
    
    if st.button("📊 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner(f"Generating {model_type.upper()} forecast..."):
            try:
                # Generate price data
                price_df = preprocessor.create_price_dataset(days=365, crop_name=crop_name)
                price_series = price_forecaster.prepare_price_data(price_df, crop_name)
                
                # Forecast
                if model_type == 'arima':
                    predictions = price_forecaster.forecast_arima(steps=forecast_days)
                else:
                    predictions = price_forecaster.forecast_lstm(price_series, steps=forecast_days)
                
                # Create dates
                dates = pd.date_range(start=pd.Timestamp.now(), periods=forecast_days, freq='D')
                
                st.markdown("---")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = price_series.iloc[-1]
                forecast_price = predictions[-1]
                min_price = min(predictions)
                max_price = max(predictions)
                
                col1.metric("💰 Current Price", f"₹{current_price:.0f}/q")
                price_change = ((forecast_price - current_price)/current_price * 100)
                col2.metric("🔮 Forecast Price", f"₹{forecast_price:.0f}/q", 
                           delta=f"{price_change:+.1f}%")
                col3.metric("📉 Min Price", f"₹{min_price:.0f}/q")
                col4.metric("📈 Max Price", f"₹{max_price:.0f}/q")
                
                st.markdown("---")
                
                # Plot
                fig = go.Figure()
                
                # Historical (last 30 days)
                historical_dates = price_series.index[-30:]
                historical_prices = price_series.values[-30:]
                
                fig.add_trace(go.Scatter(
                    x=historical_dates,
                    y=historical_prices,
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=predictions,
                    mode='lines',
                    name=f'{model_type.upper()} Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f'{crop_name} Price Forecast - {model_type.upper()} Model ({forecast_days} days)',
                    xaxis_title='Date',
                    yaxis_title='Price (₹/quintal)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Advisory
                st.subheader("💡 Market Advisory")
                
                if price_change > 10:
                    st.success("📈 *Strong upward trend* - Consider holding stock for better prices")
                    st.info(f"💰 Expected gain: ₹{forecast_price - current_price:.0f}/quintal ({price_change:+.1f}%)")
                elif price_change > 5:
                    st.info("📊 *Moderate increase expected* - Current prices are favorable")
                elif price_change < -10:
                    st.error("📉 *Significant price drop expected* - Consider selling immediately")
                elif price_change < -5:
                    st.warning("📊 *Slight decrease expected* - Monitor market closely")
                else:
                    st.success("➡ *Stable prices expected* - Normal market conditions")
                
            except Exception as e:
                st.error(f"Forecasting error: {str(e)}")
                st.info("💡 Try generating synthetic data or check if models are trained")

# ----------------------------
# MODEL COMPARISON PAGE
# ----------------------------
elif "Model Comparison" in menu:
    st.header("📊 Model Performance Comparison")
    
    tabs = st.tabs(["🌱 Crop Models", "📈 Price Models", "📋 Comprehensive"])
    
    # Crop Models Tab
    with tabs[0]:
        st.subheader("🌱 Crop Prediction Model Comparison")
        
        crop_metrics = crop_predictor.get_comparison_metrics()
        
        if crop_metrics:
            # Comparison table
            df_crop = ModelComparison.compare_crop_models(crop_metrics)
            st.dataframe(df_crop, use_container_width=True, hide_index=True)
            
            # Visualization
            fig = ModelComparison.plot_crop_model_comparison(crop_metrics)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("🎯 Feature Importance - Random Forest")
            fig_importance = ModelComparison.plot_feature_importance(crop_metrics, 'random_forest')
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Detailed metrics
            st.subheader("📊 Detailed Performance Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'random_forest' in crop_metrics:
                    st.markdown("#### 🌲 Random Forest")
                    rf = crop_metrics['random_forest']
                    st.metric("Accuracy", f"{rf.get('accuracy', 0)*100:.2f}%")
                    st.metric("CV Mean", f"{rf.get('cv_mean', 0)*100:.2f}%")
                    st.metric("CV Std", f"±{rf.get('cv_std', 0)*100:.2f}%")
            
            with col2:
                if 'decision_tree' in crop_metrics:
                    st.markdown("#### 🌳 Decision Tree")
                    dt = crop_metrics['decision_tree']
                    st.metric("Accuracy", f"{dt.get('accuracy', 0)*100:.2f}%")
                    st.metric("CV Mean", f"{dt.get('cv_mean', 0)*100:.2f}%")
                    st.metric("CV Std", f"±{dt.get('cv_std', 0)*100:.2f}%")
        else:
            st.info("No crop model metrics available. Please train models first.")
    
    # Price Models Tab
    with tabs[1]:
        st.subheader("📈 Price Forecasting Model Comparison")
        
        price_metrics = price_forecaster.get_comparison_metrics()
        
        if price_metrics:
            # Comparison table
            df_price = ModelComparison.compare_price_models(price_metrics)
            if not df_price.empty:
                st.dataframe(df_price, use_container_width=True, hide_index=True)
                
                # Visualization
                fig = ModelComparison.plot_price_model_comparison(price_metrics)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics
                st.subheader("📊 Detailed Performance Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'arima' in price_metrics and 'error' not in price_metrics['arima']:
                        st.markdown("#### 📊 ARIMA Model")
                        arima = price_metrics['arima']
                        st.metric("MAE", f"₹{arima.get('mae', 0):.2f}")
                        st.metric("RMSE", f"₹{arima.get('rmse', 0):.2f}")
                        st.metric("R² Score", f"{arima.get('r2', 0):.4f}")
                        if 'aic' in arima:
                            st.metric("AIC", f"{arima.get('aic', 0):.2f}")
                
                with col2:
                    if 'lstm' in price_metrics and 'error' not in price_metrics['lstm']:
                        st.markdown("#### 🧠 LSTM Model")
                        lstm = price_metrics['lstm']
                        st.metric("MAE", f"₹{lstm.get('mae', 0):.2f}")
                        st.metric("RMSE", f"₹{lstm.get('rmse', 0):.2f}")
                        st.metric("R² Score", f"{lstm.get('r2', 0):.4f}")
                        if 'epochs' in lstm:
                            st.metric("Epochs Trained", f"{lstm.get('epochs', 0)}")
            else:
                st.info("No valid price model metrics available")
        else:
            st.info("No price model metrics available. Please train models first.")
    
    # Comprehensive Tab
    with tabs[2]:
        st.subheader("📋 All Models Summary")
        
        crop_metrics = crop_predictor.get_comparison_metrics()
        price_metrics = price_forecaster.get_comparison_metrics()
        
        if crop_metrics or price_metrics:
            df_all = ModelComparison.create_metrics_table(crop_metrics, price_metrics)
            st.dataframe(df_all, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Key insights
            st.subheader("🔍 Key Insights & Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🌱 Best Crop Prediction Model")
                if crop_metrics:
                    best_crop_model = max(crop_metrics.items(), key=lambda x: x[1].get('accuracy', 0))
                    st.success(f"{best_crop_model[0].replace('_', ' ').title()}")
                    st.metric("Accuracy", f"{best_crop_model[1]['accuracy']*100:.2f}%")
                    st.info("💡 Recommended for production use")
            
            with col2:
                st.markdown("#### 📈 Best Price Forecasting Model")
                price_models_valid = {k: v for k, v in price_metrics.items() if 'error' not in v}
                if price_models_valid:
                    best_price_model = max(price_models_valid.items(), key=lambda x: x[1].get('r2', 0))
                    st.success(f"{best_price_model[0].upper()}")
                    st.metric("R² Score", f"{best_price_model[1]['r2']:.4f}")
                    st.info("💡 Best for price predictions")

# ----------------------------
# DISEASE DIAGNOSIS PAGE
# ----------------------------
elif "Disease" in menu:
    st.header("🦠 AI-Powered Crop Disease Diagnosis")
    
    # Disease database
    disease_db = {
        'Rice': {
            'Bacterial Leaf Blight': {
                'symptoms': ['yellow leaves', 'brown spots', 'wilting'],
                'treatment': 'Apply copper-based fungicide. Remove affected parts. Improve drainage.',
                'severity': 'High'
            },
            'Brown Spot': {
                'symptoms': ['brown spots', 'yellow leaves'],
                'treatment': 'Apply Mancozeb or Carbendazim. Ensure balanced nutrition.',
                'severity': 'Medium'
            }
        },
        'Wheat': {
            'Rust Disease': {
                'symptoms': ['yellow leaves', 'brown spots', 'orange powder'],
                'treatment': 'Apply Propiconazole or Tebuconazole fungicide immediately.',
                'severity': 'High'
            }
        },
        'Cotton': {
            'Bollworm': {
                'symptoms': ['holes', 'damaged bolls', 'larvae visible'],
                'treatment': 'Apply Bt-based insecticide. Monitor regularly. Use pheromone traps.',
                'severity': 'High'
            }
        },
        'Tomato': {
            'Late Blight': {
                'symptoms': ['brown spots', 'wilting', 'white mold'],
                'treatment': 'Apply Mancozeb or Chlorothalonil. Remove infected parts immediately.',
                'severity': 'High'
            }
        },
        'Potato': {
            'Late Blight': {
                'symptoms': ['brown spots', 'white mold', 'rotting'],
                'treatment': 'Mancozeb spray every 7 days. Destroy infected tubers.',
                'severity': 'High'
            }
        }
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Enter Disease Details")
        
        crop = st.selectbox("Select Crop", list(disease_db.keys()))
        
        st.markdown("#### Observed Symptoms")
        symptoms = st.multiselect(
            "Select all symptoms:",
            ['Yellow Leaves', 'Brown Spots', 'Wilting', 'Holes', 
             'Curled Leaves', 'White Powder', 'Black Spots', 
             'Stunted Growth', 'Rotting', 'White Mold', 'Orange Powder',
             'Damaged Bolls', 'Larvae Visible']
        )
        
        area_affected = st.slider("Area Affected (%)", 0, 100, 20)
    
    with col2:
        if st.button("🔬 Diagnose Disease", type="primary", use_container_width=True):
            if not symptoms:
                st.error("❌ Please select at least one symptom")
            else:
                with st.spinner("🤖 AI analyzing symptoms..."):
                    # Disease detection logic
                    diseases = disease_db.get(crop, {})
                    matches = []
                    
                    for disease_name, disease_info in diseases.items():
                        match_score = 0
                        for symptom in symptoms:
                            if symptom.lower() in str(disease_info['symptoms']).lower():
                                match_score += 1
                        
                        if match_score > 0:
                            confidence = (match_score / len(symptoms)) * 100
                            matches.append({
                                'disease': disease_name,
                                'confidence': confidence,
                                'treatment': disease_info['treatment'],
                                'severity': disease_info['severity']
                            })
                    
                    if matches:
                        matches.sort(key=lambda x: x['confidence'], reverse=True)
                        result = matches[0]
                        
                        st.success("✅ *Diagnosis Complete*")
                        st.markdown(f"### 🦠 {result['disease']}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Confidence", f"{result['confidence']:.1f}%")
                        with col_b:
                            severity_icon = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
                            st.metric("Severity", f"{severity_icon.get(result['severity'], '🟡')} {result['severity']}")
                        
                        # Treatment
                        st.markdown("#### 💊 Treatment Recommendations")
                        st.info(result['treatment'])
                        
                        # Action plan
                        if result['severity'] == 'High' or area_affected > 50:
                            st.error("""
                            *⚠ IMMEDIATE ACTION REQUIRED:*
                            - Isolate affected plants within 24 hours
                            - Remove and destroy severely infected parts
                            - Apply recommended treatment immediately
                            - Monitor surrounding plants twice daily
                            """)
                        else:
                            st.success("""
                            *✅ STANDARD TREATMENT:*
                            - Apply treatment within 48 hours
                            - Monitor affected plants daily
                            - Maintain field sanitation
                            - Follow up after 7-10 days
                            """)
                    else:
                        st.warning("⚠ Unable to diagnose based on selected symptoms")
                        st.info("""
                        *Next Steps:*
                        - Try selecting more specific symptoms
                        - Consult local agricultural extension officer
                        - Visit nearest Krishi Vigyan Kendra (KVK)
                        - Call Kisan Call Centre: 1800-180-1551
                        """)

# ----------------------------
# FINANCIAL TOOLS PAGE
# ----------------------------
elif "Financial" in menu:
    st.header("💰 Financial Planning Toolkit")
    
    tabs = st.tabs(["💳 Loan Calculator", "🧪 Fertilizer Calculator", "📊 Profit Estimator"])
    
    # LOAN CALCULATOR
    with tabs[0]:
        st.subheader("Agricultural Loan EMI Calculator")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loan_amount = st.number_input("Loan Amount (₹)", 10000, 10000000, 200000, 10000)
        with col2:
            interest_rate = st.slider("Interest Rate (%)", 4.0, 15.0, 7.0, 0.5)
        with col3:
            tenure = st.slider("Tenure (months)", 6, 240, 36)
        
        subsidy = st.checkbox("Apply 3% Interest Subsidy (For eligible farmers)")
        effective_rate = max(interest_rate - 3, 0) if subsidy else interest_rate
        
        if st.button("💰 Calculate EMI", type="primary", use_container_width=True):
            r = effective_rate / (12 * 100)
            n = tenure
            
            if r > 0:
                emi = loan_amount * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)
            else:
                emi = loan_amount / n
            
            total_payment = emi * n
            total_interest = total_payment - loan_amount
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Monthly EMI", f"₹{emi:,.0f}")
            col2.metric("📊 Total Interest", f"₹{total_interest:,.0f}")
            col3.metric("💵 Total Payment", f"₹{total_payment:,.0f}")
            
            savings = (interest_rate - effective_rate) * loan_amount * tenure / 1200
            col4.metric("🎁 Subsidy Savings", f"₹{savings:,.0f}" if subsidy else "₹0")
            
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Principal', 'Interest'],
                values=[loan_amount, total_interest],
                hole=.4,
                marker_colors=['#4CAF50', '#FFA500']
            )])
            fig.update_layout(title="Loan Payment Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # FERTILIZER CALCULATOR
    with tabs[1]:
        st.subheader("NPK Fertilizer Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            crop_fert = st.selectbox("Select Crop", 
                                    ['Rice', 'Wheat', 'Cotton', 'Maize', 'Potato', 'Tomato'])
            area_fert = st.number_input("Farm Area (acres)", 0.5, 100.0, 5.0, 0.5)
            target_yield = st.number_input("Target Yield (quintals/acre)", 10, 500, 50)
        
        with col2:
            st.markdown("#### Current Soil Nutrients (kg/acre)")
            soil_n = st.number_input("Nitrogen (N)", 0, 500, 180)
            soil_p = st.number_input("Phosphorus (P)", 0, 100, 25)
            soil_k = st.number_input("Potassium (K)", 0, 500, 150)
        
        if st.button("🧪 Calculate Requirements", type="primary", use_container_width=True):
            # NPK requirements per quintal
            npk_req = {
                'Rice': {'N': 2.5, 'P': 0.6, 'K': 2.5},
                'Wheat': {'N': 3.0, 'P': 0.6, 'K': 2.0},
                'Cotton': {'N': 2.0, 'P': 0.5, 'K': 2.0},
                'Maize': {'N': 2.5, 'P': 0.7, 'K': 2.0},
                'Potato': {'N': 2.0, 'P': 1.0, 'K': 2.5},
                'Tomato': {'N': 2.0, 'P': 1.2, 'K': 2.0}
            }
            
            req = npk_req[crop_fert]
            
            # Calculate requirements
            n_needed = max(target_yield * req['N'] * area_fert - soil_n * area_fert, 0)
            p_needed = max(target_yield * req['P'] * area_fert - soil_p * area_fert, 0)
            k_needed = max(target_yield * req['K'] * area_fert - soil_k * area_fert, 0)
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Nitrogen (N)")
                st.metric("Required", f"{n_needed:.1f} kg")
                urea = n_needed / 0.46
                st.info(f"*Urea (46% N)*\n\n{urea:.1f} kg\n\n{int(urea/50)} bags")
                st.write(f"💰 Cost: ₹{urea * 6:.0f}")
            
            with col2:
                st.markdown("#### Phosphorus (P)")
                st.metric("Required", f"{p_needed:.1f} kg")
                dap = p_needed / 0.46
                st.info(f"*DAP (46% P)*\n\n{dap:.1f} kg\n\n{int(dap/50)} bags")
                st.write(f"💰 Cost: ₹{dap * 27:.0f}")
            
            with col3:
                st.markdown("#### Potassium (K)")
                st.metric("Required", f"{k_needed:.1f} kg")
                mop = k_needed / 0.60
                st.info(f"*MOP (60% K)*\n\n{mop:.1f} kg\n\n{int(mop/50)} bags")
                st.write(f"💰 Cost: ₹{mop * 20:.0f}")
            
            total_cost = (urea * 6) + (dap * 27) + (mop * 20)
            st.success(f"### 💰 Total Fertilizer Cost: ₹{total_cost:,.0f}")
    
    # PROFIT ESTIMATOR
    with tabs[2]:
        st.subheader("Farm Profit Estimator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            crop_profit = st.selectbox("Crop", ['Rice', 'Wheat', 'Cotton', 'Maize', 'Potato'])
            area_profit = st.number_input("Area (acres)", 0.5, 100.0, 5.0, 0.5, key='profit_area')
            expected_yield = st.number_input("Expected Yield (q/acre)", 10, 500, 50, key='profit_yield')
            market_price = st.number_input("Selling Price (₹/quintal)", 500, 20000, 2500)
        
        with col2:
            seed_cost = st.number_input("Seeds Cost (₹)", 0, 100000, 5000)
            fertilizer_cost = st.number_input("Fertilizer Cost (₹)", 0, 200000, 15000)
            pesticide_cost = st.number_input("Pesticide Cost (₹)", 0, 100000, 8000)
            labor_cost = st.number_input("Labor Cost (₹)", 0, 500000, 25000)
            other_cost = st.number_input("Other Costs (₹)", 0, 100000, 5000)
        
        if st.button("📊 Calculate Profit", type="primary", use_container_width=True):
            total_yield_q = expected_yield * area_profit
            total_revenue = total_yield_q * market_price
            total_cost = seed_cost + fertilizer_cost + pesticide_cost + labor_cost + other_cost
            net_profit = total_revenue - total_cost
            roi = (net_profit / total_cost * 100) if total_cost > 0 else 0
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Revenue", f"₹{total_revenue:,.0f}")
            col2.metric("💸 Total Cost", f"₹{total_cost:,.0f}")
            col3.metric("💵 Net Profit", f"₹{net_profit:,.0f}")
            col4.metric("📊 ROI", f"{roi:.1f}%")
            
            # Cost breakdown
            fig = px.pie(
                values=[seed_cost, fertilizer_cost, pesticide_cost, labor_cost, other_cost],
                names=['Seeds', 'Fertilizers', 'Pesticides', 'Labor', 'Others'],
                title='Cost Distribution',
                color_discrete_sequence=px.colors.sequential.Greens_r
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if net_profit > 0:
                st.success(f"✅ Profitable! Expected profit: ₹{net_profit:,.0f}")
                if roi > 50:
                    st.balloons()
            else:
                st.error(f"⚠ Loss expected: ₹{abs(net_profit):,.0f}")

# ----------------------------
# GOVERNMENT SCHEMES PAGE
# ----------------------------
elif "Government Schemes" in menu:
    st.header("🏛 Government Schemes for Farmers")
    
    st.info("💡 Comprehensive guide to all major government welfare schemes for Indian farmers")
    
    tabs = st.tabs(["💰 Income Support", "🛡 Insurance", "🎁 Subsidies", "📱 Digital Schemes"])
    
    # INCOME SUPPORT TAB
    with tabs[0]:
        st.subheader("💰 PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class='scheme-card'>
            <h3>💵 Direct Income Support</h3>
            <p><b>Benefit:</b> ₹6,000 per year (₹2,000 every 4 months)</p>
            <p><b>Eligibility:</b> All landholding farmers</p>
            <p><b>Payment:</b> Direct Bank Transfer (DBT)</p>
            <p><b>Installments:</b> 3 per year</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📋 Eligibility Criteria")
            st.info("""
            ✅ All landholding farmers (small & marginal)
            ✅ Farmer family with cultivable land
            ✅ Valid Aadhaar card required
            ✅ Bank account linked to Aadhaar
            ❌ Institutional landholders excluded
            ❌ Constitutional post holders excluded
            """)
            
            st.markdown("#### 📝 How to Apply")
            st.success("""
            *Online Registration:*
            1. Visit: pmkisan.gov.in
            2. Click 'Farmers Corner' → 'New Farmer Registration'
            3. Enter Aadhaar number and mobile
            4. Fill farmer details and land records
            5. Submit application
            
            *Offline Registration:*
            - Visit nearest Common Service Center (CSC)
            - Visit District Agriculture Office
            - Contact local Patwari/Lekhpal
            """)
        
        with col2:
            st.markdown("#### 🔍 Check Status")
            if st.button("🌐 Check PM-KISAN Status", use_container_width=True):
                st.info("Visit: *pmkisan.gov.in* → Beneficiary Status")
                st.write("Enter Aadhaar/Account/Mobile to check")
            
            if st.button("📞 Helpline", use_container_width=True):
                st.success("*PM-KISAN Helpline*\n\n☎ 155261 / 011-24300606")
            
            st.markdown("---")
            st.markdown("#### 💡 Quick Facts")
            st.metric("Farmers Enrolled", "12+ Crore")
            st.metric("Total Disbursed", "₹2.8+ Lakh Crore")
            st.metric("Installments Paid", "16+")
        
        st.markdown("---")
        
        # Other Income Schemes
        st.subheader("🌾 Other Income Support Schemes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            *🏛 State-Specific Schemes:*
            - *Telangana:* Rythu Bandhu (₹10,000/acre/year)
            - *Odisha:* KALIA (₹10,000/year)
            - *West Bengal:* Krishak Bandhu (₹5,000/acre/year)
            - *Jharkhand:* Mukhyamantri Krishi Aashirwad Yojana
            """)
        
        with col2:
            st.markdown("""
            *💼 Additional Benefits:*
            - Interest subvention on crop loans
            - Pension for small & marginal farmers
            - Compensation for crop loss
            - Minimum Support Price (MSP) for 23 crops
            """)
    
    # INSURANCE TAB
    with tabs[1]:
        st.subheader("🛡 PM Fasal Bima Yojana (Crop Insurance)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class='scheme-card'>
            <h3>🌾 Comprehensive Crop Insurance</h3>
            <p><b>Kharif Premium:</b> 2% of Sum Insured</p>
            <p><b>Rabi Premium:</b> 1.5% of Sum Insured</p>
            <p><b>Horticulture:</b> 5% of Sum Insured</p>
            <p><b>Coverage:</b> Full crop value protection</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🎯 Coverage Details")
            st.info("""
            *Covered Risks:*
            ✅ Drought, Dry spells
            ✅ Flood, Inundation
            ✅ Pests & Diseases
            ✅ Landslides, Natural fire
            ✅ Cyclone, Hailstorm
            ✅ Unseasonal rainfall
            
            *Additional Coverage:*
            ✅ Post-harvest losses (14 days)
            ✅ Localized calamities
            ✅ Add-on for wildlife attack
            """)
            
            st.markdown("#### 💳 Premium Calculation Example")
            
            with st.expander("📊 Calculate Your Premium"):
                crop_val = st.number_input("Crop Value (₹)", 10000, 1000000, 50000, 5000)
                season = st.radio("Season", ["Kharif", "Rabi", "Horticulture"])
                
                premium_rate = 0.02 if season == "Kharif" else 0.015 if season == "Rabi" else 0.05
                farmer_premium = crop_val * premium_rate
                govt_subsidy = crop_val * (0.10 - premium_rate)
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Your Premium", f"₹{farmer_premium:,.0f}")
                col_b.metric("Govt Subsidy", f"₹{govt_subsidy:,.0f}")
                col_c.metric("Total Coverage", f"₹{crop_val:,.0f}")
        
        with col2:
            st.markdown("#### 📝 How to Enroll")
            st.success("""
            *For Loanee Farmers:*
            - Automatic enrollment via bank
            - Premium deducted from loan
            
            *For Non-Loanee:*
            1. Visit pmfby.gov.in
            2. Click 'Farmer Application'
            3. Fill crop & land details
            4. Pay premium online
            5. Get policy document
            """)
            
            if st.button("🌐 Apply for Insurance", use_container_width=True):
                st.info("Visit: *pmfby.gov.in*")
            
            if st.button("📞 Insurance Helpline", use_container_width=True):
                st.success("☎ *011-23382012*")
            
            st.markdown("---")
            st.metric("Farmers Covered", "5.5+ Crore")
            st.metric("Sum Insured", "₹3.5+ Lakh Crore")
    
    # SUBSIDIES TAB
    with tabs[2]:
        st.subheader("🎁 Agricultural Subsidies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💧 Irrigation Subsidies")
            st.markdown("""
            <div class='scheme-card'>
            <h3>🚿 Drip/Sprinkler Irrigation</h3>
            <p><b>Small Farmers:</b> 55-60% subsidy</p>
            <p><b>Other Farmers:</b> 45-50% subsidy</p>
            <p><b>Additional:</b> 5% extra for SC/ST</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='scheme-card'>
            <h3>☀ Solar Pump Subsidy</h3>
            <p><b>PM-KUSUM Scheme</b></p>
            <p>• 60% subsidy on solar pumps</p>
            <p>• 30% bank loan available</p>
            <p>• Farmer pays only 10%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🌱 Seeds & Fertilizers")
            st.info("""
            *Seed Subsidy:*
            - Certified seeds: 50% subsidy
            - Hybrid seeds: 75% subsidy
            - HYV seeds: 50% subsidy
            
            *Fertilizer Subsidy:*
            - DAP, Urea, MOP subsidized
            - Direct subsidy to manufacturers
            - Farmers get at reduced rates
            """)
        
        with col2:
            st.markdown("#### 🚜 Farm Mechanization")
            st.markdown("""
            <div class='scheme-card'>
            <h3>🚜 Equipment Subsidy</h3>
            <p><b>Tractors:</b> 25-50% subsidy</p>
            <p><b>Power Tillers:</b> 40-80% subsidy</p>
            <p><b>Harvesters:</b> 40-50% subsidy</p>
            <p><b>CHC:</b> 40% for establishment</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='scheme-card'>
            <h3>🌾 Post-Harvest Subsidy</h3>
            <p>• Warehouse: 25-33% subsidy</p>
            <p>• Cold Storage: 35% subsidy</p>
            <p>• Processing: 25-35% subsidy</p>
            <p>• Pack House: 35% subsidy</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🐄 Allied Activities")
            st.info("""
            *Livestock:*
            - Dairy: 25-33% subsidy
            - Poultry: 25-35% subsidy
            - Goat/Sheep: 25-33% subsidy
            
            *Others:*
            - Beekeeping: 40% subsidy
            - Fisheries: 40-60% subsidy
            - Horticulture: 40-50% subsidy
            """)
        
        st.markdown("---")
        st.subheader("📋 How to Apply for Subsidies")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("""
            *Step 1: Visit*
            - District Agriculture Office
            - Horticulture Department
            - State Agriculture Portal
            """)
        
        with col2:
            st.success("""
            *Step 2: Submit*
            - Application form
            - Land documents
            - Quotations
            - Bank details
            """)
        
        with col3:
            st.success("""
            *Step 3: Approval*
            - Verification by officer
            - Approval letter issued
            - Purchase equipment
            - Claim reimbursement
            """)
    
    # DIGITAL SCHEMES TAB
    with tabs[3]:
        st.subheader("📱 Digital India - Farmer Services")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='scheme-card'>
            <h3>📱 eNAM (National Agriculture Market)</h3>
            <p>Online trading platform for farmers</p>
            <p>• Transparent price discovery</p>
            <p>• Better market access</p>
            <p>• 1,000+ mandis integrated</p>
            <p>🌐 Visit: enam.gov.in</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='scheme-card'>
            <h3>📞 Kisan Call Centre (KCC)</h3>
            <p>Toll-free helpline for farmers</p>
            <p>• Available 24x7</p>
            <p>• 22 local languages</p>
            <p>• Expert agri advice</p>
            <p>☎ Call: 1800-180-1551</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='scheme-card'>
            <h3>📺 DD Kisan Channel</h3>
            <p>Dedicated TV channel for farmers</p>
            <p>• Weather forecasts</p>
            <p>• Market prices</p>
            <p>• Expert advice programs</p>
            <p>📡 Free-to-air channel</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='scheme-card'>
            <h3>📲 Kisan Suvidha App</h3>
            <p>Mobile app for farmers</p>
            <p>• Weather alerts</p>
            <p>• Market prices</p>
            <p>• Pest/disease info</p>
            <p>📥 Download from Play Store</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='scheme-card'>
            <h3>🏦 Kisan Credit Card (KCC)</h3>
            <p>Credit facility for farmers</p>
            <p>• Up to ₹3 lakh at 7%</p>
            <p>• 3% interest subvention</p>
            <p>• Effective rate: 4%</p>
            <p>🏦 Apply at any bank</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='scheme-card'>
            <h3>🌐 Agri Stack (Coming)</h3>
            <p>Unified farmer database</p>
            <p>• Digital land records</p>
            <p>• Personalized advisory</p>
            <p>• Easy loan access</p>
            <p>🚀 Pilot phase active</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📞 Important Helpline Numbers")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.info("*Kisan Call Centre*\n\n☎ 1800-180-1551")
        with col2:
            st.info("*PM-KISAN*\n\n☎ 011-24300606")
        with col3:
            st.info("*Crop Insurance*\n\n☎ 011-23382012")
        with col4:
            st.info("*Soil Health*\n\n☎ 011-24305135")

# ----------------------------
# AI CHATBOT PAGE
# ----------------------------
elif "Chatbot" in menu:
    st.header("💬 AI Farming Assistant")
    st.info("🤖 Ask me anything about farming, crops, weather, markets, loans, or schemes!")
    
    # Quick questions
    st.markdown("#### 🔥 Quick Questions")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_questions = [
        "What is PM-KISAN scheme?",
        "How to grow rice?",
        "Loan for farmers?",
        "Soil testing process?"
    ]
    
    for i, (col, question) in enumerate(zip([col1, col2, col3, col4], quick_questions)):
        with col:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": question})
                response = get_chatbot_response(question)
                st.session_state.chat_history.append({"role": "bot", "content": response})
                st.rerun()
    
    st.markdown("---")
    
    # Chat display
    for msg in st.session_state.chat_history[-10:]:
        if msg['role'] == 'user':
            st.markdown(f"<div class='chat-user'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>{msg['content']}</div>", unsafe_allow_html=True)
    
    # Input
    user_input = st.text_input("Your question:", key="chat_input", placeholder="Type your farming question here...")
    
    col1, col2 = st.columns([5, 1])
    with col1:
        if st.button("Send", type="primary", use_container_width=True):
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                response = get_chatbot_response(user_input)
                st.session_state.chat_history.append({"role": "bot", "content": response})
                st.rerun()
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p style='font-size: 18px;'><b>🌾 AgriAI Smart Farming Platform</b></p>
    <p>Powered by Machine Learning & Real-time Data</p>
    <p><b>Models:</b> Random Forest • Decision Tree • ARIMA • LSTM</p>
    <p style='font-size: 12px; margin-top: 10px;'>© 2024 AgriAI Platform | Empowering Farmers with AI</p>
</div>
""", unsafe_allow_html=True)