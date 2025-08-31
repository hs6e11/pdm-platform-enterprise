#!/usr/bin/env python3
"""
Production ML Pipeline for PDM Platform v2.0
Real-time anomaly detection with model versioning and performance monitoring
"""

import asyncio
import asyncpg
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import json
import logging
import os
import pickle
import joblib
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib

# ML Libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# Deep Learning (optional - install with: pip install tensorflow)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM models will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    TREND = "trend"
    THRESHOLD = "threshold"
    ML_DETECTED = "ml_detected"

class ModelType(Enum):
    ISOLATION_FOREST = "isolation_forest"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    STATISTICAL = "statistical"
    ENSEMBLE = "ensemble"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AnomalyResult:
    id: str
    equipment_id: str
    tenant_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AlertSeverity
    confidence: float
    anomaly_score: float
    features: Dict[str, float]
    model_version: str
    model_type: ModelType
    description: str
    recommendations: List[str]
    metadata: Dict[str, Any]

@dataclass
class ModelMetrics:
    model_id: str
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    training_samples: int
    validation_samples: int
    last_updated: datetime
    version: str
    performance_trend: str
    feature_importance: Dict[str, float]

@dataclass
class ProcessingStats:
    processed_readings: int = 0
    anomalies_detected: int = 0
    models_updated: int = 0
    processing_time: float = 0.0
    error_count: int = 0
    last_processing: datetime = None

class FeatureEngineer:
    """Advanced feature engineering for predictive maintenance"""
    
    def __init__(self):
        self.feature_history = {}
        self.baseline_stats = {}
        
    def engineer_features(self, df: pd.DataFrame, equipment_id: str = None) -> pd.DataFrame:
        """Comprehensive feature engineering pipeline"""
        if df.empty:
            return df
        
        logger.info(f"Engineering features for {equipment_id or 'all equipment'}")
        
        # Sort by timestamp
        df = df.sort_values(['equipment_id', 'timestamp']).copy()
        
        # Basic cleaning
        df = self._clean_data(df)
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Statistical features
        df = self._add_statistical_features(df)
        
        # Rolling window features
        df = self._add_rolling_features(df)
        
        # Frequency domain features
        df = self._add_frequency_features(df)
        
        # Physics-based features
        df = self._add_physics_features(df)
        
        # Anomaly indicators
        df = self._add_anomaly_indicators(df)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data cleaning and preprocessing"""
        # Handle missing values
        sensor_cols = [col for col in df.columns if col not in ['equipment_id', 'tenant_id', 'timestamp', 'is_anomaly']]
        
        # Forward fill then backward fill within each equipment
        df[sensor_cols] = df.groupby('equipment_id')[sensor_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Fill remaining NaN with column median
        for col in sensor_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Remove extreme outliers (beyond 5 sigma)
        for col in sensor_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                lower_bound = mean_val - 5 * std_val
                upper_bound = mean_val + 5 * std_val
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night_shift'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        sensor_cols = [col for col in df.columns if col in ['vibration_x', 'vibration_y', 'vibration_z', 'temperature', 'pressure', 'speed_rpm', 'current_draw', 'power_consumption']]
        
        for col in sensor_cols:
            if col not in df.columns:
                continue
                
            # Z-score (standardized values)
            df[f'{col}_zscore'] = df.groupby('equipment_id')[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
            
            # Percentile ranks
            df[f'{col}_percentile'] = df.groupby('equipment_id')[col].transform(lambda x: x.rank(pct=True))
            
            # Distance from median
            df[f'{col}_median_dist'] = df.groupby('equipment_id')[col].transform(lambda x: abs(x - x.median()))
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features"""
        sensor_cols = [col for col in df.columns if col in ['vibration_x', 'vibration_y', 'vibration_z', 'temperature', 'pressure', 'speed_rpm', 'current_draw', 'power_consumption']]
        
        # Multiple window sizes
        windows = [3, 6, 12, 24]  # Hours
        
        for window in windows:
            for col in sensor_cols:
                if col not in df.columns:
                    continue
                    
                # Rolling statistics
                rolling = df.groupby('equipment_id')[col].rolling(window=window, min_periods=1)
                df[f'{col}_roll_mean_{window}h'] = rolling.mean().values
                df[f'{col}_roll_std_{window}h'] = rolling.std().fillna(0).values
                df[f'{col}_roll_min_{window}h'] = rolling.min().values
                df[f'{col}_roll_max_{window}h'] = rolling.max().values
                
                # Rolling trends
                df[f'{col}_roll_trend_{window}h'] = rolling.apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0).values
                
                # Rolling volatility
                df[f'{col}_roll_volatility_{window}h'] = rolling.apply(lambda x: x.std() / (x.mean() + 1e-8) if len(x) > 1 else 0).values
        
        return df
    
    def _add_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency domain features using FFT"""
        vibration_cols = ['vibration_x', 'vibration_y', 'vibration_z']
        
        for equipment_id in df['equipment_id'].unique():
            equipment_data = df[df['equipment_id'] == equipment_id]
            
            if len(equipment_data) < 32:  # Need minimum samples for FFT
                continue
            
            for col in vibration_cols:
                if col not in df.columns:
                    continue
                
                try:
                    # Apply FFT to vibration data
                    signal = equipment_data[col].values
                    fft = np.fft.fft(signal)
                    freqs = np.fft.fftfreq(len(signal))
                    
                    # Extract frequency domain features
                    magnitude = np.abs(fft)
                    power_spectrum = magnitude ** 2
                    
                    # Dominant frequency
                    dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
                    dominant_freq = abs(freqs[dominant_freq_idx])
                    
                    # Spectral centroid
                    spectral_centroid = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / (np.sum(magnitude[:len(magnitude)//2]) + 1e-8)
                    
                    # RMS frequency
                    rms_freq = np.sqrt(np.mean(power_spectrum))
                    
                    # Assign back to dataframe
                    mask = df['equipment_id'] == equipment_id
                    df.loc[mask, f'{col}_dominant_freq'] = dominant_freq
                    df.loc[mask, f'{col}_spectral_centroid'] = spectral_centroid
                    df.loc[mask, f'{col}_rms_freq'] = rms_freq
                    
                except Exception as e:
                    logger.warning(f"FFT failed for {equipment_id} {col}: {str(e)}")
        
        return df
    
    def _add_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add physics-based engineering features"""
        
        # Vibration magnitude
        vib_cols = ['vibration_x', 'vibration_y', 'vibration_z']
        if all(col in df.columns for col in vib_cols):
            df['vibration_magnitude'] = np.sqrt(df[vib_cols].pow(2).sum(axis=1))
            df['vibration_rms'] = np.sqrt(df[vib_cols].pow(2).mean(axis=1))
        
        # Power efficiency
        if 'power_consumption' in df.columns and 'speed_rpm' in df.columns:
            df['power_efficiency'] = df['power_consumption'] / (df['speed_rpm'] + 1)
            df['specific_power'] = df['power_consumption'] / (df['speed_rpm'] * df.get('torque', 1) + 1)
        
        # Temperature gradients
        if 'temperature' in df.columns:
            df['temp_rate_change'] = df.groupby('equipment_id')['temperature'].diff().fillna(0)
            df['temp_acceleration'] = df.groupby('equipment_id')['temp_rate_change'].diff().fillna(0)
        
        # Pressure ratios
        if 'pressure' in df.columns:
            df['pressure_normalized'] = df.groupby('equipment_id')['pressure'].transform(lambda x: x / (x.mean() + 1e-8))
        
        # Current to speed ratio (motor health indicator)
        if 'current_draw' in df.columns and 'speed_rpm' in df.columns:
            df['current_speed_ratio'] = df['current_draw'] / (df['speed_rpm'] + 1)
        
        return df
    
    def _add_anomaly_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add anomaly detection indicators"""
        sensor_cols = [col for col in df.columns if col in ['vibration_x', 'vibration_y', 'vibration_z', 'temperature', 'pressure', 'speed_rpm', 'current_draw', 'power_consumption']]
        
        # Mahalanobis distance
        for equipment_id in df['equipment_id'].unique():
            equipment_mask = df['equipment_id'] == equipment_id
            equipment_data = df[equipment_mask][sensor_cols].select_dtypes(include=[np.number])
            
            if len(equipment_data) > len(sensor_cols):  # Need more samples than features
                try:
                    cov_matrix = np.cov(equipment_data.T)
                    mean_vector = equipment_data.mean().values
                    
                    mahal_dist = []
                    for _, row in equipment_data.iterrows():
                        diff = row.values - mean_vector
                        distance = np.sqrt(diff.T @ np.linalg.pinv(cov_matrix) @ diff)
                        mahal_dist.append(distance)
                    
                    df.loc[equipment_mask, 'mahalanobis_distance'] = mahal_dist
                except Exception as e:
                    logger.warning(f"Mahalanobis distance failed for {equipment_id}: {str(e)}")
                    df.loc[equipment_mask, 'mahalanobis_distance'] = 0
        
        # Local outlier factor approximation
        for col in sensor_cols:
            if col in df.columns:
                # Simple isolation score based on statistical deviation
                df[f'{col}_isolation_score'] = df.groupby('equipment_id')[col].transform(
                    lambda x: np.abs(x - x.median()) / (x.mad() + 1e-8)
                )
        
        return df

class ModelManager:
    """Advanced ML model management with versioning and performance tracking"""
    
    def __init__(self, models_path: str = './models'):
        self.models_path = models_path
        self.models = {}
        self.model_metrics = {}
        self.feature_engineer = FeatureEngineer()
        self.scalers = {}
        self.model_registry = {}
        
        os.makedirs(models_path, exist_ok=True)
        
    async def train_isolation_forest(self, X: np.ndarray, y: np.ndarray, equipment_id: str) -> Tuple[Any, ModelMetrics]:
        """Train Isolation Forest with hyperparameter optimization"""
        logger.info(f"Training Isolation Forest for {equipment_id}")
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Hyperparameter optimization (simplified)
        best_score = -1
        best_model = None
        contamination_rates = [0.05, 0.1, 0.15, 0.2]
        n_estimators_options = [100, 200, 300]
        
        for contamination in contamination_rates:
            for n_estimators in n_estimators_options:
                model = IsolationForest(
                    contamination=contamination,
                    n_estimators=n_estimators,
                    random_state=42,
                    max_samples='auto',
                    max_features=1.0,
                    n_jobs=-1
                )
                
                # Train model
                model.fit(X_scaled)
                
                # Simple validation using anomaly scores
                scores = model.decision_function(X_scaled)
                predictions = model.predict(X_scaled)
                
                # Score based on separation of anomalies
                anomaly_scores = scores[predictions == -1]
                normal_scores = scores[predictions == 1]
                
                if len(anomaly_scores) > 0 and len(normal_scores) > 0:
                    separation = np.mean(normal_scores) - np.mean(anomaly_scores)
                    if separation > best_score:
                        best_score = separation
                        best_model = model
        
        if best_model is None:
            # Fallback to default parameters
            best_model = IsolationForest(
                contamination=0.1,
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
            best_model.fit(X_scaled)
        
        # Calculate metrics
        predictions = best_model.predict(X_scaled)
        scores = best_model.decision_function(X_scaled)
        
        # Convert to binary classification format
        y_pred = (predictions == -1).astype(int)
        
        if len(np.unique(y)) > 1:
            try:
                auc_score = roc_auc_score(y, -scores)  # Negative scores for AUC
            except:
                auc_score = 0.5
        else:
            auc_score = 0.5
        
        metrics = ModelMetrics(
            model_id=f"{equipment_id}_isolation_forest",
            model_type=ModelType.ISOLATION_FOREST,
            accuracy=np.mean(y_pred == y) if len(np.unique(y)) > 1 else 0.0,
            precision=0.0,  # Would need proper validation set
            recall=0.0,     # Would need proper validation set
            f1_score=0.0,   # Would need proper validation set
            auc_score=auc_score,
            training_samples=len(X),
            validation_samples=0,
            last_updated=datetime.now(timezone.utc),
            version="1.0",
            performance_trend="stable",
            feature_importance={}
        )
        
        return (best_model, scaler), metrics
    
    async def train_lstm_autoencoder(self, X_sequences: np.ndarray, equipment_id: str) -> Tuple[Any, ModelMetrics]:
        """Train LSTM Autoencoder for sequence anomaly detection"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for LSTM training")
        
        logger.info(f"Training LSTM Autoencoder for {equipment_id}")
        
        # Prepare data
        X_sequences = np.nan_to_num(X_sequences, nan=0.0)
        
        # Split data
        X_train, X_val = train_test_split(X_sequences, test_size=0.2, random_state=42)
        
        # Build model
        model = self._build_lstm_autoencoder(X_sequences.shape[1:])
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        # Train model
        history = model.fit(
            X_train, X_train,  # Autoencoder reconstructs input
            epochs=100,
            batch_size=32,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=0
        )
        
        # Calculate reconstruction errors for metrics
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_mse = np.mean(np.square(X_train - train_pred))
        val_mse = np.mean(np.square(X_val - val_pred))
        
        metrics = ModelMetrics(
            model_id=f"{equipment_id}_lstm_autoencoder",
            model_type=ModelType.LSTM_AUTOENCODER,
            accuracy=1.0 - (val_mse / (train_mse + 1e-8)),  # Relative performance
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            auc_score=0.0,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            last_updated=datetime.now(timezone.utc),
            version="1.0",
            performance_trend="stable",
            feature_importance={}
        )
        
        return model, metrics
    
    def _build_lstm_autoencoder(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM Autoencoder architecture"""
        # Encoder
        input_layer = Input(shape=input_shape)
        encoded = LSTM(128, activation='relu', return_sequences=True)(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = LSTM(64, activation='relu', return_sequences=True)(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = LSTM(32, activation='relu', return_sequences=False)(encoded)
        
        # Decoder
        decoded = Dense(32, activation='relu')(encoded)
        decoded = tf.keras.layers.RepeatVector(input_shape[0])(decoded)
        decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
        decoded = Dense(input_shape[1])(decoded)
        
        # Create model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder
    
    async def train_ensemble_model(self, X: np.ndarray, y: np.ndarray, equipment_id: str) -> Tuple[Any, ModelMetrics]:
        """Train ensemble model combining multiple approaches"""
        logger.info(f"Training Ensemble Model for {equipment_id}")
        
        # Train individual models
        isolation_model, isolation_metrics = await self.train_isolation_forest(X, y, equipment_id)
        
        # Statistical model (simple threshold-based)
        statistical_model = self._create_statistical_model(X, y)
        
        # Ensemble wrapper
        ensemble_model = {
            'isolation_forest': isolation_model[0],
            'scaler': isolation_model[1],
            'statistical': statistical_model,
            'weights': {'isolation': 0.7, 'statistical': 0.3}
        }
        
        # Combined metrics
        ensemble_metrics = ModelMetrics(
            model_id=f"{equipment_id}_ensemble",
            model_type=ModelType.ENSEMBLE,
            accuracy=isolation_metrics.accuracy * 0.9,  # Ensemble typically improves
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            auc_score=isolation_metrics.auc_score,
            training_samples=len(X),
            validation_samples=0,
            last_updated=datetime.now(timezone.utc),
            version="1.0",
            performance_trend="stable",
            feature_importance={}
        )
        
        return ensemble_model, ensemble_metrics
    
    def _create_statistical_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Create statistical anomaly detection model"""
        # Calculate statistical thresholds for each feature
        thresholds = {}
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            mean_val = np.mean(feature_data)
            std_val = np.std(feature_data)
            
            thresholds[f'feature_{i}'] = {
                'mean': mean_val,
                'std': std_val,
                'lower_threshold': mean_val - 3 * std_val,
                'upper_threshold': mean_val + 3 * std_val
            }
        
        return {
            'type': 'statistical',
            'thresholds': thresholds,
            'global_threshold': np.percentile(np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)), 95)
        }
    
    async def save_model(self, model: Any, model_id: str, metrics: ModelMetrics):
        """Save model with versioning"""
        model_path = os.path.join(self.models_path, f"{model_id}_v{metrics.version}")
        
        try:
            if metrics.model_type == ModelType.LSTM_AUTOENCODER and TENSORFLOW_AVAILABLE:
                model.save(f"{model_path}.h5")
            elif metrics.model_type == ModelType.ENSEMBLE:
                # Save ensemble components separately
                joblib.dump(model, f"{model_path}.pkl")
            else:
                joblib.dump(model, f"{model_path}.pkl")
            
            # Save metrics
            with open(f"{model_path}_metrics.json", 'w') as f:
                json.dump(asdict(metrics), f, default=str, indent=2)
            
            logger.info(f"Model saved: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {str(e)}")
    
    async def load_model(self, model_id: str, version: str = "latest") -> Tuple[Any, ModelMetrics]:
        """Load model with version support"""
        if version == "latest":
            # Find latest version
            model_files = [f for f in os.listdir(self.models_path) if f.startswith(model_id)]
            if not model_files:
                raise FileNotFoundError(f"No models found for {model_id}")
            
            version_nums = []
            for f in model_files:
                try:
                    v = f.split('_v')[1].split('.')[0]
                    version_nums.append(float(v))
                except:
                    continue
            
            if not version_nums:
                raise FileNotFoundError(f"No valid versions found for {model_id}")
            
            version = str(max(version_nums))
        
        model_path = os.path.join(self.models_path, f"{model_id}_v{version}")
        
        try:
            # Load metrics first
            with open(f"{model_path}_metrics.json", 'r') as f:
                metrics_dict = json.load(f)
                metrics = ModelMetrics(**metrics_dict)
            
            # Load model based on type
            if metrics.model_type == ModelType.LSTM_AUTOENCODER and TENSORFLOW_AVAILABLE:
                model = tf.keras.models.load_model(f"{model_path}.h5")
            else:
                model = joblib.load(f"{model_path}.pkl")
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise

class ProductionMLPipeline:
    """Production ML Pipeline with real-time processing capabilities"""
    
    def __init__(self, db_url: str, models_path: str = './models'):
        self.db_url = db_url
        self.models_path = models_path
        self.model_manager = ModelManager(models_path)
        self.processing_stats = ProcessingStats()
        
        # Processing configuration
        self.batch_size = 1000
        self.processing_interval = 300  # 5 minutes
        self.model_update_interval = 86400  # 24 hours
        self.sequence_length = 24  # Hours for LSTM
        
        # Feature configuration
        self.sensor_columns = [
            'vibration_x', 'vibration_y', 'vibration_z',
            'temperature', 'pressure', 'speed_rpm',
            'current_draw', 'power_consumption'
        ]
        
        # Running state
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def connect_database(self) -> asyncpg.Connection:
        """Connect to PostgreSQL database"""
        return await asyncpg.connect(self.db_url)
    
    async def load_training_data(self, equipment_id: str = None, hours_back: int = 720) -> pd.DataFrame:
        """Load training data with comprehensive preprocessing"""
        logger.info(f"Loading training data for {equipment_id or 'all equipment'} ({hours_back}h back)")
        
        conn = await self.connect_database()
        
        try:
            query = """
                SELECT 
                    sr.equipment_id,
                    sr.sensor_type,
                    sr.value,
                    sr.timestamp,
                    sr.tenant_id::text as tenant_id,
                    CASE 
                        WHEN a.id IS NOT NULL THEN 1 
                        ELSE 0 
                    END as is_anomaly
                FROM sensor_readings sr
                LEFT JOIN anomalies a ON (
                    sr.equipment_id = a.equipment_id 
                    AND sr.timestamp BETWEEN a.detected_at - INTERVAL '5 minutes' 
                    AND a.detected_at + INTERVAL '5 minutes'
                )
                WHERE sr.timestamp >= NOW() - INTERVAL '%s hours'
                AND sr.value IS NOT NULL
            """ % hours_back
            
            params = []
            if equipment_id:
                query += " AND sr.equipment_id = $1"
                params.append(equipment_id)
            
            query += " ORDER BY sr.timestamp"
            
            rows = await conn.fetch(query, *params)
            
            if not rows:
                logger.warning("No training data found")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(row) for row in rows])
            
            # Pivot sensor data
            pivot_df = df.pivot_table(
                index=['equipment_id', 'tenant_id', 'timestamp'],
                columns='sensor_type',
                values='value',
                aggfunc='mean'
            ).reset_index()
            
            # Add anomaly labels
            anomaly_df = df.groupby(['equipment_id', 'tenant_id', 'timestamp'])['is_anomaly'].max().reset_index()
            
            # Merge
            training_data = pivot_df.merge(
                anomaly_df,
                on=['equipment_id', 'tenant_id', 'timestamp'],
                how='left'
            )
            
            training_data['is_anomaly'] = training_data['is_anomaly'].fillna(0)
            
            logger.info(f"Loaded {len(training_data)} training samples")
            return training_data
            
        finally:
            await conn.close()
    
    async def train_models(self, equipment_id: str = None, retrain: bool = False) -> Dict[str, ModelMetrics]:
        """Train ML models for equipment"""
        logger.info(f"Training models for {equipment_id or 'all equipment'}")
        
        # Load training data
        df = await self.load_training_data(equipment_id, hours_back=720)
        
        if df.empty:
            logger.warning("No training data available")
            return {}
        
        # Feature engineering
        df_engineered = self.model_manager.feature_engineer.engineer_features(df, equipment_id)
        
        # Get equipment list
        equipment_list = [equipment_id] if equipment_id else df_engineered['equipment_id'].unique()
        
        trained_models = {}
        
        for eq_id in equipment_list:
            logger.info(f"Training models for equipment: {eq_id}")
            
            equipment_data = df_engineered[df_engineered['equipment_id'] == eq_id].copy()
            
            if len(equipment_data) < 100:
                logger.warning(f"Insufficient data for {eq_id}: {len(equipment_data)} samples")
                continue
            
            # Prepare features
            feature_cols = [col for col in equipment_data.columns 
                          if col not in ['equipment_id', 'tenant_id', 'timestamp', 'is_anomaly']]
            
            X = equipment_data[feature_cols].values
            y = equipment_data['is_anomaly'].values
            
            try:
                # Train Isolation Forest
                isolation_model, isolation_metrics = await self.model_manager.train_isolation_forest(X, y, eq_id)
                await self.model_manager.save_model(isolation_model, f"{eq_id}_isolation_forest", isolation_metrics)
                trained_models[f"{eq_id}_isolation_forest"] = isolation_metrics
                
                # Train Ensemble Model
                ensemble_model, ensemble_metrics = await self.model_manager.train_ensemble_model(X, y, eq_id)
                await self.model_manager.save_model(ensemble_model, f"{eq_id}_ensemble", ensemble_metrics)
                trained_models[f"{eq_id}_ensemble"] = ensemble_metrics
                
                # Train LSTM if sufficient sequential data
                if TENSORFLOW_AVAILABLE and len(equipment_data) >= self.sequence_length * 10:
                    sequences = self._prepare_sequences(equipment_data, feature_cols)
                    if sequences.shape[0] > 20:
                        lstm_model, lstm_metrics = await self.model_manager.train_lstm_autoencoder(sequences, eq_id)
                        await self.model_manager.save_model(lstm_model, f"{eq_id}_lstm", lstm_metrics)
                        trained_models[f"{eq_id}_lstm"] = lstm_metrics
                
                self.processing_stats.models_updated += 1
                
            except Exception as e:
                logger.error(f"Failed to train models for {eq_id}: {str(e)}")
        
        logger.info(f"Training completed for {len(trained_models)} models")
        return trained_models
    
    def _prepare_sequences(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Prepare sequences for LSTM training"""
        sequences = []
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        feature_data = df_sorted[feature_cols].values
        
        # Create overlapping sequences
        for i in range(len(feature_data) - self.sequence_length + 1):
            sequence = feature_data[i:(i + self.sequence_length)]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    async def predict_anomalies(self, equipment_id: str, hours_back: int = 1) -> List[AnomalyResult]:
        """Predict anomalies using all available models"""
        logger.info(f"Predicting anomalies for {equipment_id}, last {hours_back} hours")
        
        # Load recent data
        df = await self.load_training_data(equipment_id, hours_back)
        
        if df.empty:
            return []
        
        # Feature engineering
        df_engineered = self.model_manager.feature_engineer.engineer_features(df, equipment_id)
        
        # Prepare features
        feature_cols = [col for col in df_engineered.columns 
                      if col not in ['equipment_id', 'tenant_id', 'timestamp', 'is_anomaly']]
        
        X = df_engineered[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        
        results = []
        
        try:
            # Load and apply models
            model_predictions = {}
            
            # Isolation Forest
            try:
                isolation_model, isolation_metrics = await self.model_manager.load_model(f"{equipment_id}_isolation_forest")
                if isinstance(isolation_model, tuple):
                    model, scaler = isolation_model
                    X_scaled = scaler.transform(X)
                    scores = model.decision_function(X_scaled)
                    predictions = model.predict(X_scaled)
                    model_predictions['isolation_forest'] = (scores, predictions)
            except:
                logger.warning(f"Could not load isolation forest model for {equipment_id}")
            
            # Ensemble Model
            try:
                ensemble_model, ensemble_metrics = await self.model_manager.load_model(f"{equipment_id}_ensemble")
                ensemble_scores = self._predict_ensemble(X, ensemble_model)
                model_predictions['ensemble'] = ensemble_scores
            except:
                logger.warning(f"Could not load ensemble model for {equipment_id}")
            
            # Process predictions
            for i, (_, row) in enumerate(df_engineered.iterrows()):
                anomaly_detected = False
                combined_score = 0.0
                confidence = 0.0
                model_votes = []
                
                # Combine model predictions
                for model_name, pred_data in model_predictions.items():
                    if model_name == 'isolation_forest':
                        scores, predictions = pred_data
                        if i < len(predictions):
                            is_anomaly = predictions[i] == -1
                            score = scores[i]
                            model_votes.append(is_anomaly)
                            combined_score += score
                    elif model_name == 'ensemble':
                        if i < len(pred_data):
                            is_anomaly, score = pred_data[i]
                            model_votes.append(is_anomaly)
                            combined_score += score
                
                # Majority voting
                if model_votes:
                    anomaly_detected = sum(model_votes) >= len(model_votes) / 2
                    confidence = sum(model_votes) / len(model_votes)
                    combined_score = combined_score / len(model_predictions)
                
                if anomaly_detected or abs(combined_score) > 0.3:  # Include borderline cases
                    # Determine severity
                    if abs(combined_score) > 0.7:
                        severity = AlertSeverity.CRITICAL
                    elif abs(combined_score) > 0.5:
                        severity = AlertSeverity.HIGH
                    elif abs(combined_score) > 0.3:
                        severity = AlertSeverity.MEDIUM
                    else:
                        severity = AlertSeverity.LOW
                    
                    # Generate recommendations
                    recommendations = self._generate_recommendations(equipment_id, row, combined_score)
                    
                    result = AnomalyResult(
                        id=str(uuid.uuid4()),
                        equipment_id=equipment_id,
                        tenant_id=row['tenant_id'],
                        timestamp=row['timestamp'],
                        anomaly_type=AnomalyType.ML_DETECTED,
                        severity=severity,
                        confidence=confidence,
                        anomaly_score=combined_score,
                        features={col: float(row[col]) for col in feature_cols if col in row and pd.notna(row[col])},
                        model_version="2.0.1",
                        model_type=ModelType.ENSEMBLE,
                        description=f"ML-detected anomaly in {equipment_id} with confidence {confidence:.2f}",
                        recommendations=recommendations,
                        metadata={
                            'model_votes': model_votes,
                            'feature_count': len(feature_cols),
                            'processing_time': time.time()
                        }
                    )
                    
                    results.append(result)
        
        except Exception as e:
            logger.error(f"Error in anomaly prediction for {equipment_id}: {str(e)}")
        
        # Store anomalies in database
        if results:
            await self._store_anomalies(results)
        
        logger.info(f"Anomaly prediction completed. Found {len(results)} anomalies")
        return results
    
    def _predict_ensemble(self, X: np.ndarray, ensemble_model: Dict) -> List[Tuple[bool, float]]:
        """Make predictions using ensemble model"""
        results = []
        
        # Get individual model predictions
        isolation_forest = ensemble_model['isolation_forest']
        scaler = ensemble_model['scaler']
        statistical_model = ensemble_model['statistical']
        weights = ensemble_model['weights']
        
        X_scaled = scaler.transform(X)
        
        # Isolation Forest predictions
        iso_scores = isolation_forest.decision_function(X_scaled)
        iso_predictions = isolation_forest.predict(X_scaled)
        
        # Statistical predictions
        stat_predictions = []
        stat_scores = []
        
        for i in range(len(X)):
            feature_anomalies = 0
            total_features = X.shape[1]
            
            for j, (feature_name, thresholds) in enumerate(statistical_model['thresholds'].items()):
                if j < X.shape[1]:
                    value = X[i, j]
                    if value < thresholds['lower_threshold'] or value > thresholds['upper_threshold']:
                        feature_anomalies += 1
            
            anomaly_ratio = feature_anomalies / total_features
            stat_scores.append(anomaly_ratio)
            stat_predictions.append(anomaly_ratio > 0.3)
        
        # Combine predictions
        for i in range(len(X)):
            iso_score = iso_scores[i]
            iso_pred = iso_predictions[i] == -1
            stat_score = stat_scores[i]
            stat_pred = stat_predictions[i]
            
            # Weighted combination
            combined_score = (weights['isolation'] * iso_score + 
                            weights['statistical'] * (stat_score - 0.5)) / sum(weights.values())
            
            combined_prediction = (weights['isolation'] * iso_pred + 
                                 weights['statistical'] * stat_pred) / sum(weights.values()) > 0.5
            
            results.append((combined_prediction, combined_score))
        
        return results
    
    def _generate_recommendations(self, equipment_id: str, row: pd.Series, anomaly_score: float) -> List[str]:
        """Generate contextual maintenance recommendations"""
        recommendations = []
        
        # Vibration-based recommendations
        vib_cols = ['vibration_x', 'vibration_y', 'vibration_z']
        if any(col in row.index for col in vib_cols):
            if any(abs(row.get(col, 0)) > 10 for col in vib_cols if col in row.index):
                recommendations.extend([
                    "Check bearing condition and lubrication",
                    "Inspect for mechanical looseness",
                    "Verify shaft alignment"
                ])
        
        # Temperature-based recommendations
        if 'temperature' in row.index and row['temperature'] > 80:
            recommendations.extend([
                "Check cooling system operation",
                "Inspect for blockages in air flow",
                "Verify thermal protection settings"
            ])
        
        # Current-based recommendations
        if 'current_draw' in row.index and row['current_draw'] > 50:
            recommendations.extend([
                "Check motor load conditions",
                "Inspect electrical connections",
                "Verify motor winding condition"
            ])
        
        # Speed-based recommendations
        if 'speed_rpm' in row.index:
            expected_speed = 1800  # Typical motor speed
            if abs(row['speed_rpm'] - expected_speed) > 100:
                recommendations.extend([
                    "Check drive system operation",
                    "Verify speed control settings",
                    "Inspect coupling condition"
                ])
        
        # Severity-based recommendations
        if abs(anomaly_score) > 0.7:
            recommendations.append("🚨 URGENT: Schedule immediate inspection")
            recommendations.append("Consider emergency shutdown if conditions worsen")
        elif abs(anomaly_score) > 0.5:
            recommendations.append("Schedule maintenance within 24-48 hours")
        else:
            recommendations.append("Monitor closely and schedule routine maintenance")
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "Perform routine inspection",
                "Check operational parameters",
                "Review maintenance schedule"
            ]
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    async def _store_anomalies(self, anomalies: List[AnomalyResult]):
        """Store anomalies in database"""
        conn = await self.connect_database()
        
        try:
            for anomaly in anomalies:
                await conn.execute(
                    """
                    INSERT INTO anomalies (
                        id, tenant_id, equipment_id, anomaly_type, severity,
                        confidence_score, detected_at, description, recommendations,
                        model_version, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    anomaly.id,
                    anomaly.tenant_id,
                    anomaly.equipment_id,
                    anomaly.anomaly_type.value,
                    anomaly.severity.value,
                    anomaly.confidence,
                    anomaly.timestamp,
                    anomaly.description,
                    json.dumps(anomaly.recommendations),
                    anomaly.model_version,
                    json.dumps(anomaly.metadata, default=str)
                )
        finally:
            await conn.close()
    
    async def run_continuous_processing(self, processing_interval: int = 300):
        """Run continuous ML processing pipeline"""
        logger.info(f"Starting continuous ML processing with {processing_interval}s intervals")
        
        self.is_running = True
        last_model_update = datetime.now(timezone.utc) - timedelta(days=1)  # Force initial training
        
        try:
            while self.is_running:
                start_time = time.time()
                
                try:
                    # Get active equipment
                    conn = await self.connect_database()
                    equipment_list = await conn.fetch(
                        "SELECT DISTINCT equipment_id FROM sensor_readings WHERE timestamp >= NOW() - INTERVAL '2 hours'"
                    )
                    await conn.close()
                    
                    # Process each equipment
                    for row in equipment_list:
                        equipment_id = row['equipment_id']
                        
                        try:
                            # Predict anomalies
                            anomalies = await self.predict_anomalies(equipment_id, hours_back=1)
                            self.processing_stats.anomalies_detected += len(anomalies)
                            
                        except Exception as e:
                            logger.error(f"Failed to process {equipment_id}: {str(e)}")
                            self.processing_stats.error_count += 1
                    
                    # Periodic model retraining
                    if datetime.now(timezone.utc) - last_model_update > timedelta(seconds=self.model_update_interval):
                        logger.info("Starting periodic model retraining...")
                        await self.train_models()
                        last_model_update = datetime.now(timezone.utc)
                    
                    # Update processing stats
                    self.processing_stats.processing_time = time.time() - start_time
                    self.processing_stats.last_processing = datetime.now(timezone.utc)
                    self.processing_stats.processed_readings += len(equipment_list)
                    
                except Exception as e:
                    logger.error(f"Error in processing cycle: {str(e)}")
                    self.processing_stats.error_count += 1
                
                # Wait for next cycle
                await asyncio.sleep(processing_interval)
                
        except KeyboardInterrupt:
            logger.info("Stopping ML processing pipeline...")
        finally:
            self.is_running = False

async def main():
    """Main ML Pipeline execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDM Platform ML Pipeline v2.0')
    parser.add_argument('--db-url', default='postgresql://pdm_user:password@localhost:5432/pdm_platform',
                        help='PostgreSQL connection string')
    parser.add_argument('--models-path', default='./models', help='Path to store ML models')
    parser.add_argument('--mode', choices=['train', 'predict', 'continuous'], default='continuous',
                        help='Operation mode')
    parser.add_argument('--equipment-id', help='Specific equipment ID to process')
    parser.add_argument('--interval', type=int, default=300, help='Processing interval in seconds')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ProductionMLPipeline(args.db_url, args.models_path)
    
    try:
        if args.mode == 'train':
            logger.info("🤖 Training ML models...")
            metrics = await pipeline.train_models(args.equipment_id)
            
            print("\n📊 Training Results:")
            print("=" * 50)
            for model_id, metric in metrics.items():
                print(f"✅ {model_id}: {metric.training_samples} samples, v{metric.version}")
                print(f"   Accuracy: {metric.accuracy:.3f}, AUC: {metric.auc_score:.3f}")
            
        elif args.mode == 'predict':
            logger.info("🔍 Running anomaly prediction...")
            if not args.equipment_id:
                logger.error("Equipment ID required for prediction mode")
                return
            
            anomalies = await pipeline.predict_anomalies(args.equipment_id, hours_back=2)
            
            print(f"\n🚨 Anomaly Detection Results for {args.equipment_id}:")
            print("=" * 60)
            for anomaly in anomalies:
                print(f"🔴 {anomaly.severity.value.upper()}: {anomaly.description}")
                print(f"   Confidence: {anomaly.confidence:.2f}, Score: {anomaly.anomaly_score:.3f}")
                print(f"   Time: {anomaly.timestamp}")
                if anomaly.recommendations:
                    print(f"   Recommendations: {', '.join(anomaly.recommendations[:2])}")
                print()
            
        elif args.mode == 'continuous':
            logger.info("🔄 Starting continuous processing mode...")
            await pipeline.run_continuous_processing(args.interval)
            
    except KeyboardInterrupt:
        logger.info("Shutting down ML Pipeline...")
    except Exception as e:
        logger.error(f"ML Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
