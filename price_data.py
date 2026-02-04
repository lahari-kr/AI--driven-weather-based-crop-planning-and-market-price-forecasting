"""
Script to generate synthetic historical price data
Run this once to create historical_prices.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_historical_prices():
    """Generate realistic historical price data for multiple crops"""
    
    crops_config = {
        'Rice': {'base': 2500, 'volatility': 0.15, 'trend': 0.02},
        'Wheat': {'base': 2200, 'volatility': 0.12, 'trend': 0.03},
        'Cotton': {'base': 6000, 'volatility': 0.25, 'trend': 0.01},
        'Maize': {'base': 1800, 'volatility': 0.18, 'trend': 0.02},
        'Sugarcane': {'base': 3000, 'volatility': 0.10, 'trend': 0.015},
        'Soybean': {'base': 4500, 'volatility': 0.22, 'trend': 0.025},
        'Groundnut': {'base': 5500, 'volatility': 0.20, 'trend': 0.02},
        'Potato': {'base': 1200, 'volatility': 0.35, 'trend': 0.01},
        'Tomato': {'base': 2000, 'volatility': 0.40, 'trend': 0.01},
        'Chickpea': {'base': 5000, 'volatility': 0.15, 'trend': 0.03}
    }
    
    days = 730  # 2 years of data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
    
    all_data = []
    
    for crop, config in crops_config.items():
        base_price = config['base']
        volatility = config['volatility']
        trend = config['trend']
        
        # Generate price series
        prices = [base_price]
        
        for i in range(1, days):
            # Seasonal component
            seasonal = 100 * np.sin(2 * np.pi * i / 365) * volatility
            
            # Trend component
            trend_component = trend * base_price * (i / 365)
            
            # Random walk
            random_change = np.random.normal(0, base_price * volatility / 10)
            
            # Calculate new price
            new_price = prices[-1] + seasonal + trend_component + random_change
            
            # Ensure price doesn't go too low
            new_price = max(new_price, base_price * 0.5)
            
            prices.append(new_price)
        
        # Create dataframe for this crop
        crop_df = pd.DataFrame({
            'date': dates,
            'crop': crop,
            'price': prices,
            'volume': np.random.randint(1000, 5000, days)
        })
        
        all_data.append(crop_df)
    
    # Combine all crops
    df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    output_path = Path('data/historical_prices.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df)} records for {len(crops_config)} crops")
    print(f"📁 Saved to: {output_path}")
    
    # Display summary
    print("\n📊 Summary:")
    for crop in crops_config.keys():
        crop_data = df[df['crop'] == crop]
        print(f"\n{crop}:")
        print(f"  Price range: ₹{crop_data['price'].min():.0f} - ₹{crop_data['price'].max():.0f}")
        print(f"  Average: ₹{crop_data['price'].mean():.0f}")
        print(f"  Records: {len(crop_data)}")

if __name__ == "__main__":
    generate_historical_prices()