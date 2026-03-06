"""
Disaster Prediction Pipeline
Orchestrates data generation, model training, and predictions
for the complete disaster prediction workflow.
"""

import numpy as np
import pandas as pd
import logging
import os
import json
from datetime import datetime
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DisasterPredictionPipeline:
    """
    End-to-end pipeline for natural disaster prediction.
    Manages model training, caching, and inference for floods and earthquakes.
    """
    
    MODEL_REGISTRY = {
        'flood_gb': 'models/flood_gradient_boosting.pkl',
        'flood_lstm': 'models/flood_lstm.pkl', 
        'flood_arima': 'models/flood_arima.pkl',
        'earthquake_gb': 'models/earthquake_gradient_boosting.pkl',
        'earthquake_reg': 'models/earthquake_regressor.pkl',
    }
    
    def __init__(self, model_dir: str = 'models', data_dir: str = 'data',
                 random_state: int = 42):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.random_state = random_state
        self.models = {}
        self.datasets = {}
        self.training_history = {}
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # ─── Data Loading ──────────────────────────────────────────────────────
    
    def load_or_generate_flood_data(self, n_samples: int = 5000, 
                                     force_regenerate: bool = False) -> pd.DataFrame:
        """Load cached or generate new flood training data"""
        cache_path = os.path.join(self.data_dir, 'flood_dataset.csv')
        
        if os.path.exists(cache_path) and not force_regenerate:
            logger.info(f"Loading cached flood data from {cache_path}")
            df = pd.read_csv(cache_path, parse_dates=['date'])
        else:
            logger.info("Generating flood dataset...")
            from src.data.data_generator import FloodDataGenerator
            gen = FloodDataGenerator(n_samples=n_samples, random_state=self.random_state)
            df = gen.generate()
            df.to_csv(cache_path, index=False)
            logger.info(f"Flood data saved to {cache_path}")
        
        self.datasets['flood'] = df
        return df
    
    def load_or_generate_earthquake_data(self, n_samples: int = 8000,
                                          force_regenerate: bool = False) -> pd.DataFrame:
        """Load cached or generate new earthquake training data"""
        cache_path = os.path.join(self.data_dir, 'earthquake_dataset.csv')
        
        if os.path.exists(cache_path) and not force_regenerate:
            logger.info(f"Loading cached earthquake data from {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            logger.info("Generating earthquake dataset...")
            from src.data.data_generator import EarthquakeDataGenerator
            gen = EarthquakeDataGenerator(n_samples=n_samples, random_state=self.random_state)
            df = gen.generate()
            df.to_csv(cache_path, index=False)
            logger.info(f"Earthquake data saved to {cache_path}")
        
        self.datasets['earthquake'] = df
        return df
    
    # ─── Model Training ────────────────────────────────────────────────────
    
    def train_flood_models(self, df: Optional[pd.DataFrame] = None,
                           n_samples: int = 5000) -> Dict:
        """Train all flood prediction models"""
        logger.info("=" * 60)
        logger.info("TRAINING FLOOD PREDICTION MODELS")
        logger.info("=" * 60)
        
        if df is None:
            df = self.load_or_generate_flood_data(n_samples=n_samples)
        
        results = {}
        
        # 1. Gradient Boosting (XGBoost)
        try:
            from src.models.flood_model import FloodGradientBoostingModel
            logger.info("Training XGBoost flood model...")
            xgb_model = FloodGradientBoostingModel(model_type='xgboost', 
                                                    random_state=self.random_state)
            metrics = xgb_model.train(df)
            self.models['flood_xgb'] = xgb_model
            results['xgboost'] = metrics
            logger.info(f"XGBoost: F1={metrics['f1_score']:.4f}, AUC={metrics['roc_auc']:.4f}")
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
        
        # 2. Gradient Boosting (LightGBM)
        try:
            from src.models.flood_model import FloodGradientBoostingModel
            logger.info("Training LightGBM flood model...")
            lgb_model = FloodGradientBoostingModel(model_type='lightgbm',
                                                    random_state=self.random_state)
            metrics = lgb_model.train(df)
            self.models['flood_lgb'] = lgb_model
            results['lightgbm'] = metrics
            logger.info(f"LightGBM: F1={metrics['f1_score']:.4f}, AUC={metrics['roc_auc']:.4f}")
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
        
        # 3. LSTM Time Series
        try:
            from src.models.flood_model import FloodTimeSeriesModel
            logger.info("Training LSTM flood model...")
            lstm_model = FloodTimeSeriesModel(sequence_length=24, lstm_units=64,
                                               random_state=self.random_state)
            metrics = lstm_model.train(df, epochs=30)
            self.models['flood_lstm'] = lstm_model
            results['lstm'] = metrics
            logger.info(f"LSTM: F1={metrics.get('f1_score', 0):.4f}, AUC={metrics.get('roc_auc', 0):.4f}")
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
        
        # 4. ARIMA River Level Forecaster
        try:
            from src.models.flood_model import RiverLevelForecaster
            logger.info("Training ARIMA river level forecaster...")
            arima_model = RiverLevelForecaster()
            if 'river_level_m' in df.columns:
                metrics = arima_model.train(df['river_level_m'])
                self.models['flood_arima'] = arima_model
                results['arima'] = metrics
                logger.info(f"ARIMA: MAE={metrics.get('mae', 0):.4f}, R²={metrics.get('r2', 0):.4f}")
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
        
        self.training_history['flood'] = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(df),
            'results': results
        }
        
        return results
    
    def train_earthquake_models(self, df: Optional[pd.DataFrame] = None,
                                 n_samples: int = 8000) -> Dict:
        """Train all earthquake risk models"""
        logger.info("=" * 60)
        logger.info("TRAINING EARTHQUAKE RISK MODELS")
        logger.info("=" * 60)
        
        if df is None:
            df = self.load_or_generate_earthquake_data(n_samples=n_samples)
        
        results = {}
        
        # 1. XGBoost Classifier
        try:
            from src.models.earthquake_model import EarthquakeGradientBoostingModel
            logger.info("Training XGBoost earthquake classifier...")
            xgb_model = EarthquakeGradientBoostingModel(model_type='xgboost',
                                                         random_state=self.random_state)
            metrics = xgb_model.train(df)
            self.models['earthquake_xgb'] = xgb_model
            results['xgboost_classifier'] = metrics
            logger.info(f"XGBoost: Accuracy={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")
        except Exception as e:
            logger.error(f"Earthquake XGBoost training failed: {e}")
        
        # 2. LightGBM Classifier
        try:
            from src.models.earthquake_model import EarthquakeGradientBoostingModel
            logger.info("Training LightGBM earthquake classifier...")
            lgb_model = EarthquakeGradientBoostingModel(model_type='lightgbm',
                                                         random_state=self.random_state)
            metrics = lgb_model.train(df)
            self.models['earthquake_lgb'] = lgb_model
            results['lightgbm_classifier'] = metrics
            logger.info(f"LightGBM: Accuracy={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")
        except Exception as e:
            logger.error(f"LightGBM classifier training failed: {e}")
        
        # 3. Risk Regressor
        try:
            from src.models.earthquake_model import EarthquakeRiskRegressor
            logger.info("Training earthquake risk regressor...")
            reg_model = EarthquakeRiskRegressor(model_type='xgboost',
                                                 random_state=self.random_state)
            metrics = reg_model.train(df)
            self.models['earthquake_reg'] = reg_model
            results['risk_regressor'] = metrics
            logger.info(f"Regressor: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
        except Exception as e:
            logger.error(f"Risk regressor training failed: {e}")
        
        # 4. Ensemble
        try:
            from src.models.earthquake_model import EarthquakeEnsembleModel
            logger.info("Training earthquake ensemble model...")
            ensemble = EarthquakeEnsembleModel(random_state=self.random_state)
            metrics = ensemble.train(df)
            self.models['earthquake_ensemble'] = ensemble
            results['ensemble'] = {'model_count': metrics['model_count']}
            logger.info(f"Ensemble: {metrics['model_count']} models trained")
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
        
        self.training_history['earthquake'] = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(df),
            'results': results
        }
        
        return results
    
    def train_all(self, flood_samples: int = 5000, earthquake_samples: int = 8000) -> Dict:
        """Train all models for all disaster types"""
        logger.info("Starting full training pipeline...")
        start_time = datetime.now()
        
        all_results = {
            'flood': self.train_flood_models(n_samples=flood_samples),
            'earthquake': self.train_earthquake_models(n_samples=earthquake_samples)
        }
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Full pipeline completed in {elapsed:.1f}s")
        logger.info(f"Total models trained: {len(self.models)}")
        
        return all_results
    
    # ─── Prediction Interface ──────────────────────────────────────────────
    
    def predict_flood(self, input_data: Dict, model_name: str = 'flood_xgb') -> Dict:
        """Predict flood risk for given input parameters"""
        if model_name not in self.models:
            available = [k for k in self.models.keys() if 'flood' in k]
            if not available:
                raise RuntimeError("No flood models trained. Call train_flood_models() first.")
            model_name = available[0]
        
        df = pd.DataFrame([input_data])
        result = self.models[model_name].predict(df)
        result['model_used'] = model_name
        result['timestamp'] = datetime.now().isoformat()
        
        return result
    
    def predict_earthquake(self, input_data: Dict, model_name: str = 'earthquake_xgb') -> Dict:
        """Predict earthquake risk for given location/geological parameters"""
        if model_name not in self.models:
            available = [k for k in self.models.keys() if 'earthquake' in k and 'reg' not in k]
            if not available:
                raise RuntimeError("No earthquake models trained. Call train_earthquake_models() first.")
            model_name = available[0]
        
        df = pd.DataFrame([input_data])
        result = self.models[model_name].predict(df)
        result['model_used'] = model_name
        result['timestamp'] = datetime.now().isoformat()
        
        return result
    
    def get_model_summary(self) -> Dict:
        """Return summary of all trained models and their metrics"""
        summary = {}
        for name, model in self.models.items():
            if hasattr(model, 'metrics') and model.metrics:
                summary[name] = {
                    'metrics': model.metrics,
                    'is_fitted': getattr(model, 'is_fitted', True)
                }
        return summary
    
    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        """Get feature importance for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        if hasattr(model, 'get_feature_importance'):
            return model.get_feature_importance()
        else:
            raise AttributeError(f"Model '{model_name}' does not support feature importance.")
    
    def forecast_river_level(self, steps: int = 24) -> np.ndarray:
        """Forecast river levels for the next N hours"""
        if 'flood_arima' not in self.models:
            raise RuntimeError("ARIMA model not trained.")
        return self.models['flood_arima'].forecast(steps=steps)


# ─── Convenience Functions ───────────────────────────────────────────────────

def quick_flood_prediction(rainfall_24h: float, river_level: float,
                            soil_moisture: float, rainfall_72h: float = None,
                            elevation: float = 50.0) -> Dict:
    """Quick flood prediction with minimal inputs"""
    if rainfall_72h is None:
        rainfall_72h = rainfall_24h * 2.8
    
    input_data = {
        'rainfall_24h_mm': rainfall_24h,
        'rainfall_72h_mm': rainfall_72h,
        'rainfall_7day_mm': rainfall_72h * 2.5,
        'river_level_m': river_level,
        'soil_moisture_pct': soil_moisture,
        'groundwater_level_m': 3.0,
        'elevation_m': elevation,
        'slope_degrees': 5.0,
        'drainage_area_km2': 100.0,
        'temperature_c': 20.0,
        'humidity_pct': min(60 + rainfall_24h * 0.5, 100),
        'wind_speed_kmh': 15.0
    }
    
    pipeline = DisasterPredictionPipeline()
    flood_df = pipeline.load_or_generate_flood_data(n_samples=3000)
    pipeline.train_flood_models(flood_df)
    return pipeline.predict_flood(input_data)


def quick_earthquake_assessment(latitude: float, longitude: float,
                                  fault_distance_km: float = 30.0,
                                  historical_earthquakes: int = 5) -> Dict:
    """Quick earthquake risk assessment for a location"""
    input_data = {
        'latitude': latitude,
        'longitude': longitude,
        'fault_distance_km': fault_distance_km,
        'fault_type': 'strike-slip',
        'plate_velocity_mm_yr': 7.0,
        'historical_earthquakes_5yr': historical_earthquakes,
        'last_major_event_years': 15.0,
        'rock_type': 'soft_soil',
        'soil_amplification_factor': 2.5,
        'depth_to_bedrock_m': 30.0,
        'vs30_m_s': 400.0,
        'population_density_km2': 500.0,
        'building_age_avg_years': 35.0,
        'seismic_code_compliance': 0.6
    }
    
    pipeline = DisasterPredictionPipeline()
    eq_df = pipeline.load_or_generate_earthquake_data(n_samples=3000)
    pipeline.train_earthquake_models(eq_df)
    return pipeline.predict_earthquake(input_data)


if __name__ == '__main__':
    print("Starting Disaster Prediction Pipeline...")
    
    pipeline = DisasterPredictionPipeline()
    results = pipeline.train_all(flood_samples=3000, earthquake_samples=3000)
    
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    
    summary = pipeline.get_model_summary()
    for model_name, data in summary.items():
        print(f"\n{model_name}:")
        metrics = data['metrics']
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif not isinstance(v, dict):
                print(f"  {k}: {v}")
