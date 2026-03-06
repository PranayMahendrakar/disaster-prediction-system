"""
Training Script for Disaster Prediction System
Run this script to train all models and evaluate performance.

Usage:
    python train.py                          # Train all models
    python train.py --disaster flood         # Train flood models only
    python train.py --disaster earthquake    # Train earthquake models only
    python train.py --model-type lightgbm    # Use LightGBM instead of XGBoost
    python train.py --flood-samples 10000    # Custom sample count
"""

import argparse
import logging
import sys
import os
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', 
                           mode='w', delay=True)
    ]
)
logger = logging.getLogger(__name__)

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Disaster Prediction Models')
    parser.add_argument('--disaster', choices=['all', 'flood', 'earthquake'], 
                       default='all', help='Which disaster type to train')
    parser.add_argument('--model-type', choices=['xgboost', 'lightgbm', 'sklearn'],
                       default='xgboost', help='Gradient boosting model type')
    parser.add_argument('--flood-samples', type=int, default=5000,
                       help='Number of flood training samples')
    parser.add_argument('--earthquake-samples', type=int, default=8000,
                       help='Number of earthquake training samples')
    parser.add_argument('--train-lstm', action='store_true',
                       help='Also train LSTM time series model')
    parser.add_argument('--train-arima', action='store_true',
                       help='Also train ARIMA river level forecaster')
    parser.add_argument('--train-ensemble', action='store_true',
                       help='Train earthquake ensemble model')
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained models to disk')
    parser.add_argument('--output-json', type=str, default=None,
                       help='Save training results to JSON file')
    return parser.parse_args()


def print_separator(title=''):
    print('=' * 60)
    if title:
        print(f'  {title}')
        print('=' * 60)


def train_flood_models(args):
    """Train flood prediction models"""
    print_separator('FLOOD PREDICTION MODELS')
    
    # Generate data
    logger.info(f"Generating flood dataset ({args.flood_samples} samples)...")
    from src.data.data_generator import FloodDataGenerator
    gen = FloodDataGenerator(n_samples=args.flood_samples, random_state=42)
    df = gen.generate()
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Flood rate: {df['flood_occurred'].mean():.2%}")
    
    results = {}
    
    # Gradient Boosting Models
    from src.models.flood_model import FloodGradientBoostingModel
    
    for mtype in [args.model_type]:
        logger.info(f"\nTraining FloodGradientBoostingModel ({mtype})...")
        model = FloodGradientBoostingModel(model_type=mtype, random_state=42)
        metrics = model.train(df)
        results[f'flood_{mtype}'] = metrics
        
        print(f"\n{mtype.upper()} Flood Model:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Feature importance
        fi = model.get_feature_importance()
        print(f"\n  Top 5 Features:")
        for _, row in fi.head(5).iterrows():
            print(f"    {row['feature']:<30} {row['importance']:.4f}")
        
        if args.save_models:
            path = f'models/flood_{mtype}.pkl'
            model.save(path)
            logger.info(f"Saved to {path}")
    
    # LSTM Model
    if args.train_lstm:
        logger.info("\nTraining LSTM flood model...")
        from src.models.flood_model import FloodTimeSeriesModel
        lstm = FloodTimeSeriesModel(sequence_length=24, lstm_units=64, random_state=42)
        lstm_metrics = lstm.train(df, epochs=30)
        results['flood_lstm'] = lstm_metrics
        print(f"\nLSTM Flood Model:")
        print(f"  Accuracy:  {lstm_metrics.get('accuracy', 0):.4f}")
        print(f"  F1 Score:  {lstm_metrics.get('f1_score', 0):.4f}")
        print(f"  ROC-AUC:   {lstm_metrics.get('roc_auc', 0):.4f}")
    
    # ARIMA Model
    if args.train_arima:
        logger.info("\nTraining ARIMA river level forecaster...")
        from src.models.flood_model import RiverLevelForecaster
        arima = RiverLevelForecaster()
        arima_metrics = arima.train(df['river_level_m'])
        results['flood_arima'] = arima_metrics
        print(f"\nARIMA River Level Forecaster:")
        print(f"  MAE:  {arima_metrics.get('mae', 0):.4f}")
        print(f"  RMSE: {arima_metrics.get('rmse', 0):.4f}")
        print(f"  R²:   {arima_metrics.get('r2', 0):.4f}")
    
    return results


def train_earthquake_models(args):
    """Train earthquake risk models"""
    print_separator('EARTHQUAKE RISK MODELS')
    
    # Generate data
    logger.info(f"Generating earthquake dataset ({args.earthquake_samples} samples)...")
    from src.data.data_generator import EarthquakeDataGenerator
    gen = EarthquakeDataGenerator(n_samples=args.earthquake_samples, random_state=42)
    df = gen.generate()
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Risk distribution: {df['risk_category'].value_counts().to_dict()}")
    
    results = {}
    
    # Classifier
    from src.models.earthquake_model import EarthquakeGradientBoostingModel
    
    for mtype in [args.model_type]:
        logger.info(f"\nTraining EarthquakeGradientBoostingModel ({mtype})...")
        model = EarthquakeGradientBoostingModel(model_type=mtype, random_state=42)
        metrics = model.train(df)
        results[f'earthquake_{mtype}'] = metrics
        
        print(f"\n{mtype.upper()} Earthquake Classifier:")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")
        print(f"  ROC-AUC OvR:  {metrics['roc_auc_ovr']:.4f}")
        
        # Feature importance
        fi = model.get_feature_importance()
        print(f"\n  Top 5 Features:")
        for _, row in fi.head(5).iterrows():
            print(f"    {row['feature']:<35} {row['importance']:.4f}")
        
        if args.save_models:
            path = f'models/earthquake_{mtype}.pkl'
            model.save(path)
            logger.info(f"Saved to {path}")
    
    # Regressor
    from src.models.earthquake_model import EarthquakeRiskRegressor
    logger.info("\nTraining EarthquakeRiskRegressor...")
    regressor = EarthquakeRiskRegressor(model_type=args.model_type, random_state=42)
    reg_metrics = regressor.train(df)
    results['earthquake_regressor'] = reg_metrics
    
    print(f"\nEarthquake Risk Regressor:")
    print(f"  MAE:  {reg_metrics['mae']:.4f}")
    print(f"  RMSE: {reg_metrics['rmse']:.4f}")
    print(f"  R²:   {reg_metrics['r2']:.4f}")
    
    # Ensemble
    if args.train_ensemble:
        logger.info("\nTraining EarthquakeEnsembleModel...")
        from src.models.earthquake_model import EarthquakeEnsembleModel
        ensemble = EarthquakeEnsembleModel(random_state=42)
        ens_metrics = ensemble.train(df)
        results['earthquake_ensemble'] = {
            'model_count': ens_metrics['model_count'],
            'weights': ens_metrics['weights']
        }
        print(f"\nEarthquake Ensemble ({ens_metrics['model_count']} models):")
        for mtype, w in ens_metrics['weights'].items():
            print(f"  {mtype}: weight={w:.3f}")
    
    return results


def main():
    args = parse_args()
    
    print_separator('DISASTER PREDICTION SYSTEM - TRAINING')
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Disaster: {args.disaster}")
    print(f"  Model type: {args.model_type}")
    print()
    
    all_results = {}
    
    try:
        if args.disaster in ('all', 'flood'):
            flood_results = train_flood_models(args)
            all_results['flood'] = flood_results
        
        if args.disaster in ('all', 'earthquake'):
            eq_results = train_earthquake_models(args)
            all_results['earthquake'] = eq_results
        
        print_separator('TRAINING COMPLETE')
        total_models = sum(len(v) for v in all_results.values())
        print(f"  Total models trained: {total_models}")
        print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if args.output_json:
            # Convert numpy types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_types(i) for i in obj]
                elif hasattr(obj, 'item'):
                    return obj.item()
                return obj
            
            with open(args.output_json, 'w') as f:
                json.dump(convert_types(all_results), f, indent=2)
            logger.info(f"Results saved to {args.output_json}")
        
        return all_results
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
