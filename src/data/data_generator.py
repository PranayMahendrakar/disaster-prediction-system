"""
Data Generator for Disaster Prediction System
Generates synthetic training data for flood and earthquake models
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class FloodDataGenerator:
    """Generates synthetic flood prediction dataset"""
    
    def __init__(self, n_samples: int = 5000, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate(self) -> pd.DataFrame:
        """Generate flood dataset with realistic features"""
        logger.info(f"Generating {self.n_samples} flood samples...")
        
        dates = pd.date_range(start='2015-01-01', periods=self.n_samples, freq='6H')
        
        # Rainfall features (mm)
        rainfall_24h = np.random.exponential(scale=15, size=self.n_samples)
        rainfall_72h = rainfall_24h * np.random.uniform(2.5, 3.5, self.n_samples)
        rainfall_7day = rainfall_72h * np.random.uniform(2, 3, self.n_samples)
        
        # River and water features
        river_level = 2.0 + rainfall_24h * 0.1 + np.random.normal(0, 0.5, self.n_samples)
        river_level = np.clip(river_level, 0, 15)
        soil_moisture = np.random.uniform(20, 95, self.n_samples)
        groundwater_level = np.random.uniform(1, 10, self.n_samples)
        
        # Topographic features
        elevation = np.random.uniform(0, 500, self.n_samples)
        slope = np.random.uniform(0, 45, self.n_samples)
        drainage_area = np.random.exponential(100, self.n_samples)
        
        # Weather features
        temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(self.n_samples) / (365 * 4)) +                       np.random.normal(0, 3, self.n_samples)
        humidity = np.clip(60 + rainfall_24h * 0.5 + np.random.normal(0, 10, self.n_samples), 0, 100)
        wind_speed = np.random.exponential(15, self.n_samples)
        
        # Seasonal factor
        month = dates.month
        seasonal_factor = 1.0 + 0.5 * np.sin(2 * np.pi * (month - 6) / 12)
        
        # Generate flood risk score (0-1)
        flood_score = (
            0.3 * (rainfall_24h / rainfall_24h.max()) +
            0.2 * (rainfall_72h / rainfall_72h.max()) +
            0.15 * (river_level / river_level.max()) +
            0.1 * (soil_moisture / 100) +
            0.1 * (1 - elevation / 500) +
            0.05 * (drainage_area / drainage_area.max()) +
            0.1 * seasonal_factor / 1.5
        )
        
        # Binary flood label with threshold
        flood_occurred = (flood_score > 0.45).astype(int)
        flood_occurred = self._add_noise(flood_occurred, noise_rate=0.03)
        
        df = pd.DataFrame({
            'date': dates,
            'rainfall_24h_mm': np.round(rainfall_24h, 2),
            'rainfall_72h_mm': np.round(rainfall_72h, 2),
            'rainfall_7day_mm': np.round(rainfall_7day, 2),
            'river_level_m': np.round(river_level, 2),
            'soil_moisture_pct': np.round(soil_moisture, 1),
            'groundwater_level_m': np.round(groundwater_level, 2),
            'elevation_m': np.round(elevation, 1),
            'slope_degrees': np.round(slope, 1),
            'drainage_area_km2': np.round(drainage_area, 1),
            'temperature_c': np.round(temperature, 1),
            'humidity_pct': np.round(humidity, 1),
            'wind_speed_kmh': np.round(wind_speed, 1),
            'flood_risk_score': np.round(flood_score, 4),
            'flood_occurred': flood_occurred
        })
        
        logger.info(f"Generated dataset: {df.shape}, Flood rate: {df['flood_occurred'].mean():.2%}")
        return df
    
    def _add_noise(self, labels: np.ndarray, noise_rate: float = 0.03) -> np.ndarray:
        """Add label noise to simulate real-world data"""
        noisy = labels.copy()
        noise_idx = np.random.choice(len(labels), size=int(len(labels) * noise_rate), replace=False)
        noisy[noise_idx] = 1 - noisy[noise_idx]
        return noisy
    
    def generate_time_series(self, n_stations: int = 5, n_timesteps: int = 1000) -> Dict[str, pd.DataFrame]:
        """Generate time-series data for multiple monitoring stations"""
        stations = {}
        for i in range(n_stations):
            dates = pd.date_range(start='2020-01-01', periods=n_timesteps, freq='1H')
            base_rainfall = np.random.exponential(5, n_timesteps)
            
            # Add temporal patterns
            hourly_pattern = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)
            seasonal_pattern = 1 + 0.5 * np.sin(2 * np.pi * np.arange(n_timesteps) / (24 * 365))
            
            rainfall = base_rainfall * hourly_pattern * seasonal_pattern
            river_level = 2 + rainfall.cumsum() * 0.001 % 10
            
            stations[f'station_{i+1}'] = pd.DataFrame({
                'timestamp': dates,
                'rainfall_mm': np.round(rainfall, 2),
                'river_level_m': np.round(river_level, 2),
                'temperature_c': np.round(20 + 5 * seasonal_pattern + np.random.normal(0, 1, n_timesteps), 1),
                'humidity_pct': np.round(np.clip(70 + rainfall * 2 + np.random.normal(0, 5, n_timesteps), 0, 100), 1)
            })
        
        return stations


class EarthquakeDataGenerator:
    """Generates synthetic earthquake risk dataset"""
    
    REGIONS = {
        'Pacific Ring of Fire': {'lat_range': (-50, 60), 'lon_range': (100, -70), 'base_risk': 0.7},
        'Alpine Himalayan Belt': {'lat_range': (20, 50), 'lon_range': (0, 100), 'base_risk': 0.6},
        'Mid-Atlantic Ridge': {'lat_range': (-60, 70), 'lon_range': (-45, -10), 'base_risk': 0.4},
        'East African Rift': {'lat_range': (-30, 15), 'lon_range': (25, 45), 'base_risk': 0.35},
        'Stable Continental': {'lat_range': (40, 60), 'lon_range': (-10, 30), 'base_risk': 0.15}
    }
    
    def __init__(self, n_samples: int = 8000, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate(self) -> pd.DataFrame:
        """Generate earthquake risk dataset"""
        logger.info(f"Generating {self.n_samples} earthquake samples...")
        
        records = []
        samples_per_region = self.n_samples // len(self.REGIONS)
        
        for region_name, region_params in self.REGIONS.items():
            for _ in range(samples_per_region):
                lat = np.random.uniform(*region_params['lat_range'])
                lon = np.random.uniform(-180, 180)
                
                # Seismic features
                fault_distance = np.random.exponential(50)
                fault_type = np.random.choice(['strike-slip', 'thrust', 'normal', 'oblique'])
                plate_velocity = np.random.uniform(1, 15)
                historical_earthquakes = np.random.poisson(region_params['base_risk'] * 20)
                last_major_event_years = np.random.exponential(30)
                
                # Geological features
                rock_type = np.random.choice(['hard_rock', 'soft_soil', 'alluvium', 'fill'])
                soil_amplification = {'hard_rock': 1.0, 'soft_soil': 3.0, 'alluvium': 2.0, 'fill': 4.0}[rock_type]
                depth_to_bedrock_m = np.random.uniform(0, 100)
                vs30 = np.random.uniform(150, 1500)  # Shear wave velocity
                
                # Infrastructure
                population_density = np.random.exponential(500)
                building_age_avg = np.random.uniform(5, 80)
                seismic_code_compliance = np.random.uniform(0, 1)
                
                # Calculate risk score
                risk_score = (
                    region_params['base_risk'] * 0.3 +
                    (1 / (1 + fault_distance / 20)) * 0.25 +
                    min(plate_velocity / 15, 1) * 0.15 +
                    min(historical_earthquakes / 40, 1) * 0.15 +
                    (1 / (1 + last_major_event_years / 50)) * 0.05 +
                    (soil_amplification / 4) * 0.1
                )
                risk_score = np.clip(risk_score + np.random.normal(0, 0.05), 0, 1)
                
                # Risk category
                if risk_score >= 0.65:
                    risk_category = 'HIGH'
                elif risk_score >= 0.35:
                    risk_category = 'MEDIUM'
                else:
                    risk_category = 'LOW'
                
                records.append({
                    'region': region_name,
                    'latitude': round(lat, 4),
                    'longitude': round(lon, 4),
                    'fault_distance_km': round(fault_distance, 2),
                    'fault_type': fault_type,
                    'plate_velocity_mm_yr': round(plate_velocity, 2),
                    'historical_earthquakes_5yr': historical_earthquakes,
                    'last_major_event_years': round(last_major_event_years, 1),
                    'rock_type': rock_type,
                    'soil_amplification_factor': soil_amplification,
                    'depth_to_bedrock_m': round(depth_to_bedrock_m, 1),
                    'vs30_m_s': round(vs30, 1),
                    'population_density_km2': round(population_density, 1),
                    'building_age_avg_years': round(building_age_avg, 1),
                    'seismic_code_compliance': round(seismic_code_compliance, 3),
                    'earthquake_risk_score': round(risk_score, 4),
                    'risk_category': risk_category
                })
        
        df = pd.DataFrame(records)
        logger.info(f"Generated dataset: {df.shape}")
        logger.info(f"Risk distribution: {df['risk_category'].value_counts().to_dict()}")
        return df


if __name__ == '__main__':
    # Test data generation
    print("Generating Flood Dataset...")
    flood_gen = FloodDataGenerator(n_samples=1000)
    flood_df = flood_gen.generate()
    print(flood_df.head())
    print(f"Flood columns: {list(flood_df.columns)}")
    
    print("\nGenerating Earthquake Dataset...")
    eq_gen = EarthquakeDataGenerator(n_samples=500)
    eq_df = eq_gen.generate()
    print(eq_df.head())
    print(f"Earthquake columns: {list(eq_df.columns)}")
