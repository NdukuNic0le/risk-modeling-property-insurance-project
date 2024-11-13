import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point
import random  

class KenyaPropertyDataGenerator:
    def __init__(self, start_date='2020-01-01', end_date='2024-01-01'):
        # Kenya's approximate bounding box
        self.lat_bounds = (-4.678589, 4.621506)
        self.lon_bounds = (33.908859, 41.899078)
        
        # Major cities coordinates for realistic property clustering
        self.major_cities = {
            'Nairobi': (-1.286389, 36.817223),
            'Mombasa': (-4.043477, 39.658871),
            'Kisumu': (-0.091702, 34.767956),
            'Nakuru': (-0.303099, 36.080026),
            'Eldoret': (0.514277, 35.269779)
        }
        
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        # Risk factors for different regions
        self.risk_zones = {
            'Coastal': {'flood_risk': 'high', 'base_risk_multiplier': 1.5},
            'Urban': {'flood_risk': 'medium', 'base_risk_multiplier': 1.2},
            'Rural': {'flood_risk': 'low', 'base_risk_multiplier': 1.0}
        }

    def generate_property_data(self, num_properties=1000):
        properties = []
        
        for _ in range(num_properties):
            # Randomly select a base location near a major city
            city, coords = random.choice(list(self.major_cities.items()))
            
            # Add some random variation to the location
            lat = coords[0] + np.random.normal(0, 0.1)
            lon = coords[1] + np.random.normal(0, 0.1)
            
            # Determine zone type based on location
            zone_type = self._determine_zone_type(lat, lon)
            
            property_data = {
                'property_id': f'PROP_{_:05d}',
                'latitude': lat,
                'longitude': lon,
                'city': city,
                'zone_type': zone_type,
                'property_type': np.random.choice(['Residential', 'Commercial', 'Industrial'], p=[0.7, 0.2, 0.1]),
                'property_value': self._generate_property_value(zone_type),
                'year_built': np.random.randint(1960, 2024),
                'elevation': self._generate_elevation(zone_type),
                'distance_to_water': self._generate_water_distance(zone_type),
                'flood_zone': self.risk_zones[zone_type]['flood_risk']
            }
            
            properties.append(property_data)
        
        return pd.DataFrame(properties)

    def generate_weather_events(self, property_data):
        weather_events = []
        
        # Generate events for each month in the date range
        current_date = self.start_date
        while current_date <= self.end_date:
            # More events during rainy seasons (March-May and October-December)
            is_rainy_season = current_date.month in [3, 4, 5, 10, 11, 12]
            
            for _, property_row in property_data.iterrows():
                if np.random.random() < (0.1 if is_rainy_season else 0.02):
                    event = {
                        'property_id': property_row['property_id'],
                        'date': current_date,
                        'event_type': np.random.choice(['Flood', 'Hail', 'Storm'], 
                                                     p=[0.5, 0.2, 0.3]),
                        'severity': np.random.choice(['Minor', 'Moderate', 'Severe'],
                                                   p=[0.5, 0.3, 0.2]),
                        'damage_value': self._generate_damage_value(property_row['property_value'])
                    }
                    weather_events.append(event)
            
            current_date += timedelta(days=30)
        
        return pd.DataFrame(weather_events)

    def _determine_zone_type(self, lat, lon):
        # Simplified zone determination
        if abs(lon - self.major_cities['Mombasa'][1]) < 1:
            return 'Coastal'
        elif any(abs(lat - city[0]) < 0.5 and abs(lon - city[1]) < 0.5 
                for city in self.major_cities.values()):
            return 'Urban'
        return 'Rural'

    def _generate_property_value(self, zone_type):
        base_values = {
            'Coastal': 10000000,  # 10M KES base
            'Urban': 8000000,     # 8M KES base
            'Rural': 5000000      # 5M KES base
        }
        return int(np.random.lognormal(
            mean=np.log(base_values[zone_type]), 
            sigma=0.5
        ))

    def _generate_elevation(self, zone_type):
        elevation_ranges = {
            'Coastal': (1, 50),
            'Urban': (1500, 2000),
            'Rural': (1000, 2500)
        }
        return np.random.uniform(*elevation_ranges[zone_type])

    def _generate_water_distance(self, zone_type):
        if zone_type == 'Coastal':
            return np.random.uniform(0, 2)
        elif zone_type == 'Urban':
            return np.random.uniform(1, 10)
        return np.random.uniform(2, 20)

    def _generate_damage_value(self, property_value):
        return int(property_value * np.random.uniform(0.01, 0.3))

    def save_to_csv(self, property_data, weather_data, output_dir='data/'):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        property_data.to_csv(f'{output_dir}kenya_property_data.csv', index=False)
        weather_data.to_csv(f'{output_dir}kenya_weather_events.csv', index=False)

# Let's test the generator
if __name__ == "__main__":
    # Create an instance of the generator
    generator = KenyaPropertyDataGenerator()
    
    # Generate property data
    property_data = generator.generate_property_data(1000)
    print("\nProperty Data Sample:")
    print(property_data.head())
    
    # Generate weather events
    weather_data = generator.generate_weather_events(property_data)
    print("\nWeather Events Sample:")
    print(weather_data.head())
    
    # Save the data
    generator.save_to_csv(property_data, weather_data)
    print("\nData has been generated and saved!")