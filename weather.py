import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np

load_dotenv()

class WeatherService:
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY', '')
        self.base_url = 'https://api.openweathermap.org/data/2.5'
        
    def get_coordinates_from_location(self, location_name):
        """Get lat/lon from location name using OpenWeather Geocoding"""
        if not self.api_key:
            return self._get_fallback_coordinates(location_name)
        
        url = f'http://api.openweathermap.org/geo/1.0/direct?q={location_name}&limit=1&appid={self.api_key}'
        try:
            response = requests.get(url, timeout=5)
            data = response.json()
            if data:
                return {
                    'lat': data[0]['lat'],
                    'lon': data[0]['lon'],
                    'name': data[0]['name'],
                    'country': data[0].get('country', '')
                }
        except Exception as e:
            print(f"Geocoding error: {e}")
        
        return self._get_fallback_coordinates(location_name)
    
    def _get_fallback_coordinates(self, location_name):
        """Fallback coordinates for major Indian cities"""
        fallback_cities = {
            'andhra_pradesh': {'lat': 16.5062, 'lon': 80.6480, 'name': 'Amaravati', 'country': 'IN'},
            'arunachal_pradesh': {'lat': 27.0844, 'lon': 93.6053, 'name': 'Itanagar', 'country': 'IN'},
            'assam': {'lat': 26.2006, 'lon': 92.9376, 'name': 'Dispur', 'country': 'IN'},
            'bihar': {'lat': 25.6110, 'lon': 85.1440, 'name': 'Patna', 'country': 'IN'},
            'chhattisgarh': {'lat': 21.2514, 'lon': 81.6296, 'name': 'Raipur', 'country': 'IN'},
            'goa': {'lat': 15.4909, 'lon': 73.8278, 'name': 'Panaji', 'country': 'IN'},
            'gujarat': {'lat': 23.0225, 'lon': 72.5714, 'name': 'Gandhinagar', 'country': 'IN'},
            'haryana': {'lat': 30.7333, 'lon': 76.7794, 'name': 'Chandigarh', 'country': 'IN'},
            'himachal_pradesh': {'lat': 31.1048, 'lon': 77.1734, 'name': 'Shimla', 'country': 'IN'},
            'jharkhand': {'lat': 23.3441, 'lon': 85.3096, 'name': 'Ranchi', 'country': 'IN'},
            'karnataka': {'lat': 12.9716, 'lon': 77.5946, 'name': 'Bengaluru', 'country': 'IN'},
            'kerala': {'lat': 8.5241, 'lon': 76.9366, 'name': 'Thiruvananthapuram', 'country': 'IN'},
            'madhya_pradesh': {'lat': 23.2599, 'lon': 77.4126, 'name': 'Bhopal', 'country': 'IN'},
            'maharashtra': {'lat': 18.5204, 'lon': 73.8567, 'name': 'Mumbai', 'country': 'IN'},
            'manipur': {'lat': 24.8170, 'lon': 93.9368, 'name': 'Imphal', 'country': 'IN'},
            'meghalaya': {'lat': 25.5788, 'lon': 91.8933, 'name': 'Shillong', 'country': 'IN'},
            'mizoram': {'lat': 23.1645, 'lon': 92.9376, 'name': 'Aizawl', 'country': 'IN'},
            'nagaland': {'lat': 25.6751, 'lon': 94.1086, 'name': 'Kohima', 'country': 'IN'},
            'odisha': {'lat': 20.2961, 'lon': 85.8245, 'name': 'Bhubaneswar', 'country': 'IN'},
            'punjab': {'lat': 30.7333, 'lon': 76.7794, 'name': 'Chandigarh', 'country': 'IN'},
            'rajasthan': {'lat': 26.9124, 'lon': 75.7873, 'name': 'Jaipur', 'country': 'IN'},
            'sikkim': {'lat': 27.3389, 'lon': 88.6065, 'name': 'Gangtok', 'country': 'IN'},
            'tamil_nadu': {'lat': 13.0827, 'lon': 80.2707, 'name': 'Chennai', 'country': 'IN'},
            'telangana': {'lat': 17.3850, 'lon': 78.4867, 'name': 'Hyderabad', 'country': 'IN'},
            'tripura': {'lat': 23.8315, 'lon': 91.2868, 'name': 'Agartala', 'country': 'IN'},
            'uttar_pradesh': {'lat': 26.8467, 'lon': 80.9462, 'name': 'Lucknow', 'country': 'IN'},
            'uttarakhand': {'lat': 30.3165, 'lon': 78.0322, 'name': 'Dehradun', 'country': 'IN'},
            'west_bengal': {'lat': 22.5726, 'lon': 88.3639, 'name': 'Kolkata', 'country': 'IN'},
            
            # Union Territories
            'andaman_nicobar': {'lat': 11.6234, 'lon': 92.7265, 'name': 'Port Blair', 'country': 'IN'},
            'chandigarh': {'lat': 30.7333, 'lon': 76.7794, 'name': 'Chandigarh', 'country': 'IN'},
            'dadra_nagar_haveli_daman_diu': {'lat': 20.3974, 'lon': 72.8328, 'name': 'Daman', 'country': 'IN'},
            'delhi': {'lat': 28.6139, 'lon': 77.2090, 'name': 'Delhi', 'country': 'IN'},
            'jammu_kashmir': {'lat': 34.0837, 'lon': 74.7973, 'name': 'Srinagar', 'country': 'IN'},
            'ladakh': {'lat': 34.1526, 'lon': 77.5770, 'name': 'Leh', 'country': 'IN'},
            'lakshadweep': {'lat': 10.5667, 'lon': 72.6417, 'name': 'Kavaratti', 'country': 'IN'},
            'puducherry': {'lat': 11.9416, 'lon': 79.8083, 'name': 'Puducherry', 'country': 'IN'},
            'bengaluru': {'lat': 12.9716, 'lon': 77.5946, 'name': 'Bengaluru', 'country': 'IN'},
             'chennai': {'lat': 13.0827, 'lon': 80.2707, 'name': 'Chennai', 'country': 'IN'},
            'hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'name': 'Hyderabad', 'country': 'IN'},
            'kochi': {'lat': 9.9312, 'lon': 76.2673, 'name': 'Kochi', 'country': 'IN'},
            'thiruvananthapuram': {'lat': 8.5241, 'lon': 76.9366, 'name': 'Thiruvananthapuram', 'country': 'IN'},
            'coimbatore': {'lat': 11.0168, 'lon': 76.9558, 'name': 'Coimbatore', 'country': 'IN'},
            'madurai': {'lat': 9.9252, 'lon': 78.1198, 'name': 'Madurai', 'country': 'IN'},
            'tiruchirappalli': {'lat': 10.7905, 'lon': 78.7047, 'name': 'Tiruchirappalli', 'country': 'IN'},
            'mangalore': {'lat': 12.9141, 'lon': 74.8560, 'name': 'Mangalore', 'country': 'IN'},
            'mysuru': {'lat': 12.2958, 'lon': 76.6394, 'name': 'Mysuru', 'country': 'IN'},
            'visakhapatnam': {'lat': 17.6868, 'lon': 83.2185, 'name': 'Visakhapatnam', 'country': 'IN'},
            'vijayawada': {'lat': 16.5062, 'lon': 80.6480, 'name': 'Vijayawada', 'country': 'IN'},
            'warangal': {'lat': 17.9784, 'lon': 79.5941, 'name': 'Warangal', 'country': 'IN'},
            'bhopal': {'lat': 23.2599, 'lon': 77.4126, 'name': 'Bhopal', 'country': 'IN'},
            'indore': {'lat': 22.7196, 'lon': 75.8577, 'name': 'Indore', 'country': 'IN'},
            'jabalpur': {'lat': 23.1815, 'lon': 79.9864, 'name': 'Jabalpur', 'country': 'IN'},
            'raipur': {'lat': 21.2514, 'lon': 81.6296, 'name': 'Raipur', 'country': 'IN'},
            'bilaspur': {'lat': 22.0797, 'lon': 82.1391, 'name': 'Bilaspur', 'country': 'IN'},
            'gwalior': {'lat': 26.2183, 'lon': 78.1828, 'name': 'Gwalior', 'country': 'IN'},

            # --- North-East & Islands ---
            'dimapur': {'lat': 25.9063, 'lon': 93.7276, 'name': 'Dimapur', 'country': 'IN'},
            'dibrugarh': {'lat': 27.4728, 'lon': 94.9120, 'name': 'Dibrugarh', 'country': 'IN'},
            'silchar': {'lat': 24.8333, 'lon': 92.7789, 'name': 'Silchar', 'country': 'IN'},
            'port_blair': {'lat': 11.6234, 'lon': 92.7265, 'name': 'Port Blair', 'country': 'IN'},
            'puducherry': {'lat': 11.9416, 'lon': 79.8083, 'name': 'Puducherry', 'country': 'IN'},
            'kavaratti': {'lat': 10.5667, 'lon': 72.6417, 'name': 'Kavaratti', 'country': 'IN'},
            'vellore': {'lat': 12.9165, 'lon': 79.1325, 'name': 'Vellore', 'country': 'IN'},
            'chittoor': {'lat': 13.2172, 'lon': 79.1003, 'name': 'Chittoor', 'country': 'IN'},

}

        
        location_lower = location_name.lower()
        for city in fallback_cities:
            if city in location_lower:
                return fallback_cities[city]
        
        # Default to Delhi
        return fallback_cities['delhi']
    
    def get_current_weather(self, lat, lon):
        """Get current weather data"""
        if not self.api_key:
            return self._generate_synthetic_weather(lat, lon)
        
        url = f'{self.base_url}/weather?lat={lat}&lon={lon}&units=metric&appid={self.api_key}'
        try:
            response = requests.get(url, timeout=5)
            data = response.json()
            
            return {
                'temp': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'clouds': data['clouds']['all']
            }
        except Exception as e:
            print(f"Weather API error: {e}")
            return self._generate_synthetic_weather(lat, lon)
    
    def get_forecast(self, lat, lon, days=7):
        """Get weather forecast"""
        if not self.api_key:
            return self._generate_synthetic_forecast(lat, lon, days)
        
        url = f'{self.base_url}/forecast?lat={lat}&lon={lon}&units=metric&appid={self.api_key}'
        try:
            response = requests.get(url, timeout=5)
            data = response.json()
            
            forecast = []
            for item in data['list'][:days*8]:  # 8 items per day (3-hour intervals)
                forecast.append({
                    'date': item['dt_txt'],
                    'temp': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'rain': item.get('rain', {}).get('3h', 0),
                    'description': item['weather'][0]['description']
                })
            
            return forecast
        except Exception as e:
            print(f"Forecast API error: {e}")
            return self._generate_synthetic_forecast(lat, lon, days)
    
    def _generate_synthetic_weather(self, lat, lon):
        """Generate realistic synthetic weather data"""
        # Climate zones based on latitude
        if lat > 30:  # Northern region
            temp_range = (15, 35)
            base_humidity = 50
        elif lat > 20:  # Central region
            temp_range = (20, 38)
            base_humidity = 60
        else:  # Southern region
            temp_range = (22, 35)
            base_humidity = 70
        
        # Seasonal adjustment
        month = datetime.now().month
        if month in [12, 1, 2]:  # Winter
            temp_mod = -5
            humidity_mod = -10
        elif month in [6, 7, 8, 9]:  # Monsoon
            temp_mod = 0
            humidity_mod = 15
        else:  # Summer
            temp_mod = 5
            humidity_mod = -5
        
        return {
            'temp': np.random.uniform(temp_range[0] + temp_mod, temp_range[1] + temp_mod),
            'humidity': np.clip(base_humidity + humidity_mod + np.random.uniform(-10, 10), 20, 100),
            'pressure': np.random.uniform(1010, 1020),
            'description': 'partly cloudy',
            'wind_speed': np.random.uniform(2, 15),
            'clouds': np.random.randint(20, 80)
        }
    
    def _generate_synthetic_forecast(self, lat, lon, days):
        """Generate synthetic forecast"""
        forecast = []
        base_weather = self._generate_synthetic_weather(lat, lon)
        
        for i in range(days):
            date = datetime.now() + timedelta(days=i)
            temp_variation = np.random.uniform(-3, 3)
            
            forecast.append({
                'date': date.strftime('%Y-%m-%d'),
                'temp_max': round(base_weather['temp'] + temp_variation + 3, 1),
                'temp_min': round(base_weather['temp'] + temp_variation - 5, 1),
                'humidity': round(base_weather['humidity'] + np.random.uniform(-10, 10), 1),
                'rain': round(np.random.uniform(0, 20), 1),
                'description': base_weather['description']
            })
        
        return forecast