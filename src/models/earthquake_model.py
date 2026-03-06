"""
Earthquake Risk Prediction Models
Implements Gradient Boosting and ensemble methods for earthquake risk assessment
with support for multi-class risk classification and regression risk scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
)
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─── Feature Definitions ────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    'fault_distance_km', 'plate_velocity_mm_yr', 'historical_earthquakes_5yr',
    'last_major_event_years', 'soil_amplification_factor', 'depth_to_bedrock_m',
    'vs30_m_s', 'population_density_km2', 'building_age_avg_years',
    'seismic_code_compliance', 'latitude', 'longitude'
]

CATEGORICAL_FEATURES = ['fault_type', 'rock_type']

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_CLASSIFICATION = 'risk_category'
TARGET_REGRESSION = 'earthquake_risk_score'


# ─── XGBoost Earthquake Classifier ──────────────────────────────────────────

class EarthquakeGradientBoostingModel:
    """
    Multi-class earthquake risk classifier using gradient boosting.
    Classifies regions into LOW / MEDIUM / HIGH risk categories.
    """
    
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.metrics = {}
        self.classes_ = None
    
    def _build_preprocessor(self):
        """Build sklearn preprocessing pipeline"""
        numeric_transformer = Pipeline([('scaler', StandardScaler())])
        categorical_transformer = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, NUMERIC_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES)
            ],
            remainder='drop'
        )
    
    def _build_classifier(self):
        """Build the gradient boosting classifier"""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=400,
                max_depth=7,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.2,
                reg_alpha=0.2,
                reg_lambda=1.5,
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=7,
                learning_rate=0.04,
                num_leaves=127,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.2,
                reg_lambda=1.5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        else:
            logger.warning("Using sklearn GradientBoosting as fallback")
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state
            )
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare dataframe"""
        missing_num = [f for f in NUMERIC_FEATURES if f not in df.columns]
        missing_cat = [f for f in CATEGORICAL_FEATURES if f not in df.columns]
        
        if missing_num or missing_cat:
            logger.warning(f"Missing features: {missing_num + missing_cat}")
        
        # Fill missing numeric with median
        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        
        # Fill missing categorical with 'unknown'
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = 'unknown'
        
        return df
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Train the earthquake risk classification model"""
        logger.info(f"Training EarthquakeGradientBoostingModel ({self.model_type})...")
        
        df = self.prepare_data(df.copy())
        
        X = df[ALL_FEATURES]
        y_raw = df[TARGET_CLASSIFICATION]
        y = self.label_encoder.fit_transform(y_raw)
        self.classes_ = self.label_encoder.classes_
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Build pipeline
        self.preprocessor = self._build_preprocessor()
        classifier = self._build_classifier()
        
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', classifier)
        ])
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        
        self.metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
            'weighted_f1': float(f1_score(y_test, y_pred, average='weighted')),
            'roc_auc_ovr': float(roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')),
            'classes': self.classes_.tolist(),
            'model_type': self.model_type,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=self.classes_,
                output_dict=True
            )
        }
        
        logger.info(f"Training complete. Accuracy: {self.metrics['accuracy']:.4f}, "
                    f"F1: {self.metrics['macro_f1']:.4f}")
        return self.metrics
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Predict earthquake risk for given locations"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        
        df = self.prepare_data(df.copy())
        X = df[ALL_FEATURES]
        
        y_pred_encoded = self.model.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_prob = self.model.predict_proba(X)
        
        # Get class probabilities
        prob_dict = {}
        for i, cls in enumerate(self.classes_):
            prob_dict[f'prob_{cls.lower()}'] = y_prob[:, i].tolist()
        
        max_prob = y_prob.max(axis=1)
        
        return {
            'risk_category': y_pred.tolist(),
            'confidence': max_prob.tolist(),
            'class_probabilities': prob_dict
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        classifier = self.model.named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        else:
            return pd.DataFrame({'feature': ALL_FEATURES, 'importance': [0.0]*len(ALL_FEATURES)})
        
        # Get feature names after preprocessing
        preprocessor = self.model.named_steps['preprocessor']
        try:
            feature_names = (
                list(NUMERIC_FEATURES) + 
                list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(CATEGORICAL_FEATURES))
            )
        except Exception:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Align lengths
        min_len = min(len(feature_names), len(importances))
        
        return pd.DataFrame({
            'feature': feature_names[:min_len],
            'importance': importances[:min_len]
        }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    def save(self, path: str):
        """Save model to disk"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'metrics': self.metrics,
            'is_fitted': self.is_fitted,
            'classes': self.classes_
        }, path)
        logger.info(f"Model saved: {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.metrics = data['metrics']
        self.is_fitted = data['is_fitted']
        self.classes_ = data['classes']
        logger.info(f"Model loaded: {path}")
        return self


# ─── Regression Risk Scorer ──────────────────────────────────────────────────

class EarthquakeRiskRegressor:
    """
    Continuous earthquake risk score regression model.
    Outputs risk scores between 0.0 (no risk) and 1.0 (maximum risk).
    """
    
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.is_fitted = False
        self.metrics = {}
    
    def _build_regressor(self):
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        else:
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=self.random_state
            )
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Train the risk regression model"""
        logger.info(f"Training EarthquakeRiskRegressor ({self.model_type})...")
        
        # Prepare features
        df_prep = df.copy()
        for col in NUMERIC_FEATURES:
            if col not in df_prep.columns:
                df_prep[col] = 0.0
        for col in CATEGORICAL_FEATURES:
            if col not in df_prep.columns:
                df_prep[col] = 'unknown'
        
        X = df_prep[ALL_FEATURES]
        y = df_prep[TARGET_REGRESSION].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Build pipeline
        numeric_transformer = Pipeline([('scaler', StandardScaler())])
        categorical_transformer = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer([
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
        
        regressor = self._build_regressor()
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', regressor)
        ])
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        y_pred = np.clip(self.model.predict(X_test), 0, 1)
        
        self.metrics = {
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'r2': float(r2_score(y_test, y_pred)),
            'model_type': self.model_type,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Training complete. R²: {self.metrics['r2']:.4f}, MAE: {self.metrics['mae']:.4f}")
        return self.metrics
    
    def predict_risk_score(self, df: pd.DataFrame) -> np.ndarray:
        """Predict continuous risk scores"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        df_prep = df.copy()
        for col in NUMERIC_FEATURES:
            if col not in df_prep.columns:
                df_prep[col] = 0.0
        for col in CATEGORICAL_FEATURES:
            if col not in df_prep.columns:
                df_prep[col] = 'unknown'
        
        scores = self.model.predict(df_prep[ALL_FEATURES])
        return np.clip(scores, 0, 1)


# ─── Ensemble Model ──────────────────────────────────────────────────────────

class EarthquakeEnsembleModel:
    """
    Ensemble combining XGBoost, LightGBM, and Random Forest
    for robust earthquake risk prediction.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.weights = {}
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.metrics = {}
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Train ensemble of models"""
        logger.info("Training EarthquakeEnsembleModel...")
        
        model_types = []
        if XGBOOST_AVAILABLE:
            model_types.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            model_types.append('lightgbm')
        model_types.append('sklearn')
        
        individual_metrics = {}
        for mtype in model_types:
            m = EarthquakeGradientBoostingModel(model_type=mtype, random_state=self.random_state)
            metrics = m.train(df, test_size=test_size)
            self.models[mtype] = m
            individual_metrics[mtype] = metrics
            logger.info(f"  {mtype}: Accuracy={metrics['accuracy']:.4f}")
        
        # Weight by accuracy
        total_acc = sum(m['accuracy'] for m in individual_metrics.values())
        self.weights = {
            mtype: individual_metrics[mtype]['accuracy'] / total_acc 
            for mtype in individual_metrics
        }
        
        self.is_fitted = True
        self.label_encoder = list(self.models.values())[0].label_encoder
        
        self.metrics = {
            'individual_metrics': individual_metrics,
            'weights': self.weights,
            'model_count': len(self.models)
        }
        
        logger.info(f"Ensemble training complete with {len(self.models)} models")
        return self.metrics
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Ensemble prediction via weighted voting"""
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted.")
        
        all_probs = []
        classes = None
        
        for mtype, model in self.models.items():
            result = model.predict(df)
            prob_df = pd.DataFrame({
                k: v for k, v in result['class_probabilities'].items()
            })
            all_probs.append(prob_df.values * self.weights[mtype])
            if classes is None:
                classes = model.classes_
        
        # Weighted average probabilities
        ensemble_probs = np.sum(all_probs, axis=0)
        pred_indices = np.argmax(ensemble_probs, axis=1)
        predictions = [classes[i] for i in pred_indices]
        confidences = ensemble_probs.max(axis=1)
        
        return {
            'risk_category': predictions,
            'confidence': confidences.tolist(),
            'ensemble_probs': ensemble_probs.tolist()
        }


if __name__ == '__main__':
    from src.data.data_generator import EarthquakeDataGenerator
    
    print("Generating Earthquake Dataset...")
    gen = EarthquakeDataGenerator(n_samples=2000)
    df = gen.generate()
    
    print("\nTraining XGBoost Earthquake Classifier...")
    classifier = EarthquakeGradientBoostingModel(model_type='xgboost')
    metrics = classifier.train(df)
    print(f"Classifier metrics: {metrics['accuracy']:.4f} accuracy, {metrics['macro_f1']:.4f} F1")
    
    print("\nTraining Risk Regressor...")
    regressor = EarthquakeRiskRegressor(model_type='xgboost')
    reg_metrics = regressor.train(df)
    print(f"Regressor metrics: R²={reg_metrics['r2']:.4f}, MAE={reg_metrics['mae']:.4f}")
    
    print("\nSample predictions:")
    sample = df.head(5)
    preds = classifier.predict(sample)
    print(f"Risk categories: {preds['risk_category']}")
    print(f"Confidence: {[f'{c:.2f}' for c in preds['confidence']]}")
