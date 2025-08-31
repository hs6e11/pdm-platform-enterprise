# ml-pipeline/orchestrator.py
"""
ML Pipeline Orchestrator for PDM Platform v2.0
Implements edge computing, anomaly detection, and cognitive maintenance
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
import pickle
import os
from pathlib import Path
from dataclasses import dataclass, asdict
import uuid
import time
from collections import defaultdict, deque
import threading
import redis

# ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
import joblib
import numpy as np

# Database connection
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AnomalyResult:
    """Anomaly detection result"""
    machine_id: str
    sensor_type: str
    timestamp: datetime
    value: float
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    model_used: str
    explanation: Dict[str, Any]

@dataclass
class MaintenancePrediction:
    """Maintenance prediction result"""
    machine_id: str
    prediction_timestamp: datetime
    failure_probability: float
    predicted_failure_time: Optional[datetime]
    time_to_failure_hours: Optional[float]
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    recommended_actions: List[str]
    confidence: float
    model_version: str

@dataclass
class CognitiveRecommendation:
    """Cognitive maintenance recommendation"""
    machine_id: str
    recommendation_id: str
    recommendation_type: str  # 'preventive', 'corrective', 'predictive'
    description: str
    priority: int  # 1-5, with 5 being highest
    estimated_cost_savings: float
    implementation_difficulty: int  # 1-5, with 5 being most difficult
    expected_completion_time: timedelta
    supporting_evidence: List[str]
    created_at: datetime

class EdgeAnomalyDetector:
    """Edge-optimized anomaly detection for real-time processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.training_data = defaultdict(lambda: deque(maxlen=1000))
        self.last_training = {}
        self.model_performance = defaultdict(dict)
        
        # Model weights for ensemble (as per whitepaper)
        self.ensemble_weights = {
            'isolation_forest': 0.4,
            'statistical': 0.2, 
            'lstm': 0.4
        }
        
        logger.info("Edge Anomaly Detector initialized")
    
    def _prepare_statistical_features(self, values: List[float], window_size: int = 10) -> Dict[str, float]:
        """Calculate statistical features for anomaly detection"""
        if len(values) < window_size:
            return {}
        
        recent_values = values[-window_size:]
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        
        return {
            'z_score': abs((values[-1] - mean_val) / max(std_val, 1e-8)),
            'moving_avg_deviation': abs(values[-1] - mean_val),
            'trend': np.polyfit(range(window_size), recent_values, 1)[0],
            'volatility': std_val / max(mean_val, 1e-8)
        }
    
    def _train_isolation_forest(self, machine_id: str, sensor_type: str, data: np.ndarray) -> bool:
        """Train Isolation Forest model for specific machine-sensor combination"""
        try:
            model_key = f"{machine_id}_{sensor_type}"
            
            # Configure model parameters
            contamination = self.config.get('contamination_rate', 0.1)
            n_estimators = self.config.get('n_estimators', 100)
            
            # Create and train model
            model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
            
            # Prepare features (current value + statistical features)
            features = []
            for i in range(len(data)):
                if i >= 10:  # Need at least 10 points for statistical features
                    stats = self._prepare_statistical_features(data[:i+1].tolist())
                    if stats:
                        feature_vector = [data[i]] + list(stats.values())
                        features.append(feature_vector)
            
            if len(features) < 50:  # Need minimum data for training
                logger.warning(f"Insufficient data for training {model_key}: {len(features)} samples")
                return False
            
            features_array = np.array(features)
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            # Train model
            model.fit(features_scaled)
            
            # Store model and scaler
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.last_training[model_key] = datetime.utcnow()
            
            # Evaluate model performance
            predictions = model.predict(features_scaled)
            anomaly_scores = model.decision_function(features_scaled)
            
            self.model_performance[model_key]['isolation_forest'] = {
                'anomaly_rate': (predictions == -1).sum() / len(predictions),
                'avg_anomaly_score': np.mean(anomaly_scores[predictions == -1]) if (predictions == -1).any() else 0,
                'training_samples': len(features),
                'last_trained': self.last_training[model_key].isoformat()
            }
            
            logger.info(f"Trained Isolation Forest for {model_key} with {len(features)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train Isolation Forest for {machine_id}.{sensor_type}: {e}")
            return False
    
    def _create_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Create lightweight LSTM model for edge deployment"""
        model = keras.Sequential([
            keras.layers.LSTM(32, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(16, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')  # Anomaly probability
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _train_lstm_model(self, machine_id: str, sensor_type: str, data: np.ndarray) -> bool:
        """Train LSTM model for temporal anomaly detection"""
        try:
            model_key = f"{machine_id}_{sensor_type}"
            sequence_length = self.config.get('lstm_sequence_length', 20)
            
            if len(data) < sequence_length * 3:  # Need enough data
                logger.warning(f"Insufficient data for LSTM training {model_key}: {len(data)} samples")
                return False
            
            # Prepare sequences
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            
            sequences = []
            labels = []
            
            # Create sequences and labels (assuming normal data for training)
            for i in range(sequence_length, len(data_scaled)):
                sequences.append(data_scaled[i-sequence_length:i])
                # Simple labeling: consider outliers as anomalies
                current_val = data_scaled[i]
                recent_mean = np.mean(data_scaled[max(0, i-50):i])
                recent_std = np.std(data_scaled[max(0, i-50):i])
                
                # Label as anomaly if value is more than 3 std devs from recent mean
                is_anomaly = abs(current_val - recent_mean) > 3 * recent_std
                labels.append(1 if is_anomaly else 0)
            
            X = np.array(sequences).reshape(-1, sequence_length, 1)
            y = np.array(labels)
            
            # Balance dataset if too few anomalies
            anomaly_ratio = np.mean(y)
            if anomaly_ratio < 0.05:  # Less than 5% anomalies
                # Add some synthetic anomalies
                normal_indices = np.where(y == 0)[0]
                num_synthetic = min(len(normal_indices) // 10, 50)
                
                for _ in range(num_synthetic):
                    idx = np.random.choice(normal_indices)
                    # Add noise to create synthetic anomaly
                    X_synthetic = X[idx].copy()
                    X_synthetic += np.random.normal(0, 0.3, X_synthetic.shape)
                    X = np.vstack([X, X_synthetic.reshape(1, -1, 1)])
                    y = np.append(y, 1)
            
            # Create and train model
            model = self._create_lstm_model((sequence_length, 1))
            
            # Train with validation split
            history = model.fit(
                X, y,
                epochs=20,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Store model and scaler
            self.models[f"{model_key}_lstm"] = model
            self.scalers[f"{model_key}_lstm"] = scaler
            
            # Evaluate performance
            predictions = model.predict(X)
            binary_predictions = (predictions > 0.5).astype(int)
            
            self.model_performance[model_key]['lstm'] = {
                'accuracy': np.mean(binary_predictions.flatten() == y),
                'anomaly_detection_rate': np.mean(binary_predictions[y == 1]) if np.any(y == 1) else 0,
                'false_positive_rate': np.mean(binary_predictions[y == 0]) if np.any(y == 0) else 0,
                'training_samples': len(X),
                'last_trained': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Trained LSTM for {model_key} with {len(X)} sequences")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train LSTM for {machine_id}.{sensor_type}: {e}")
            return False
    
    def update_training_data(self, machine_id: str, sensor_type: str, value: float, timestamp: datetime):
        """Add new data point and trigger retraining if needed"""
        key = f"{machine_id}_{sensor_type}"
        self.training_data[key].append((timestamp, value))
        
        # Check if retraining is needed
        last_training_time = self.last_training.get(key)
        if (not last_training_time or 
            (datetime.utcnow() - last_training_time).total_seconds() > 3600):  # Retrain every hour
            
            if len(self.training_data[key]) >= 100:  # Minimum data for retraining
                values = np.array([v for t, v in self.training_data[key]])
                
                # Train models in background thread to avoid blocking
                threading.Thread(
                    target=self._retrain_models,
                    args=(machine_id, sensor_type, values),
                    daemon=True
                ).start()
    
    def _retrain_models(self, machine_id: str, sensor_type: str, data: np.ndarray):
        """Retrain models for a specific machine-sensor combination"""
        logger.info(f"Retraining models for {machine_id}.{sensor_type}")
        
        # Train Isolation Forest
        self._train_isolation_forest(machine_id, sensor_type, data)
        
        # Train LSTM (more computationally expensive, so less frequent)
        if len(data) > 200:
            self._train_lstm_model(machine_id, sensor_type, data)
    
    async def detect_anomaly(self, machine_id: str, sensor_type: str, value: float, 
                           timestamp: datetime, historical_values: List[float] = None) -> AnomalyResult:
        """Detect anomalies using ensemble of models"""
        
        # Update training data
        self.update_training_data(machine_id, sensor_type, value, timestamp)
        
        model_key = f"{machine_id}_{sensor_type}"
        anomaly_scores = {}
        explanations = {}
        
        # Statistical anomaly detection
        if historical_values and len(historical_values) >= 10:
            stats = self._prepare_statistical_features(historical_values + [value])
            if stats:
                z_score = stats['z_score']
                anomaly_scores['statistical'] = min(z_score / 3.0, 1.0)  # Normalize to 0-1
                explanations['statistical'] = {
                    'z_score': z_score,
                    'threshold': 3.0,
                    'is_anomaly': z_score > 3.0
                }
        
        # Isolation Forest detection
        if model_key in self.models and model_key in self.scalers:
            try:
                # Prepare features
                if historical_values and len(historical_values) >= 10:
                    stats = self._prepare_statistical_features(historical_values + [value])
                    if stats:
                        features = np.array([[value] + list(stats.values())])
                        features_scaled = self.scalers[model_key].transform(features)
                        
                        # Get anomaly score and prediction
                        anomaly_score = self.models[model_key].decision_function(features_scaled)[0]
                        prediction = self.models[model_key].predict(features_scaled)[0]
                        
                        # Normalize anomaly score to 0-1 (lower scores mean more anomalous)
                        normalized_score = max(0, min(1, (anomaly_score + 1) / 2))  # Rough normalization
                        anomaly_scores['isolation_forest'] = 1 - normalized_score  # Invert so higher = more anomalous
                        
                        explanations['isolation_forest'] = {
                            'raw_score': anomaly_score,
                            'prediction': 'anomaly' if prediction == -1 else 'normal',
                            'normalized_score': normalized_score
                        }
            except Exception as e:
                logger.warning(f"Isolation Forest prediction failed for {model_key}: {e}")
        
        # LSTM detection
        lstm_key = f"{model_key}_lstm"
        if lstm_key in self.models and lstm_key in self.scalers:
            try:
                sequence_length = self.config.get('lstm_sequence_length', 20)
                if historical_values and len(historical_values) >= sequence_length:
                    # Prepare sequence
                    sequence_data = np.array(historical_values[-sequence_length:])
                    sequence_scaled = self.scalers[lstm_key].transform(sequence_data.reshape(-1, 1)).flatten()
                    sequence_input = sequence_scaled.reshape(1, sequence_length, 1)
                    
                    # Get prediction
                    lstm_score = self.models[lstm_key].predict(sequence_input, verbose=0)[0][0]
                    anomaly_scores['lstm'] = float(lstm_score)
                    
                    explanations['lstm'] = {
                        'anomaly_probability': float(lstm_score),
                        'threshold': 0.5,
                        'sequence_length': sequence_length
                    }
            except Exception as e:
                logger.warning(f"LSTM prediction failed for {lstm_key}: {e}")
        
        # Ensemble prediction
        if anomaly_scores:
            weighted_score = 0
            total_weight = 0
            
            for model_name, score in anomaly_scores.items():
                weight = self.ensemble_weights.get(model_name, 0.33)
                weighted_score += score * weight
                total_weight += weight
            
            final_anomaly_score = weighted_score / total_weight if total_weight > 0 else 0
            
            # Determine if it's an anomaly (configurable threshold)
            threshold = self.config.get('anomaly_threshold', 0.7)
            is_anomaly = final_anomaly_score > threshold
            
            # Calculate confidence based on model agreement
            scores_list = list(anomaly_scores.values())
            if len(scores_list) > 1:
                confidence = 1 - (np.std(scores_list) / np.mean(scores_list)) if np.mean(scores_list) > 0 else 0.5
            else:
                confidence = 0.8  # Medium confidence with single model
            
            model_used = f"ensemble_{len(anomaly_scores)}_models"
            
        else:
            # Fallback to simple statistical method
            if historical_values and len(historical_values) >= 10:
                mean_val = np.mean(historical_values[-10:])
                std_val = np.std(historical_values[-10:])
                z_score = abs(value - mean_val) / max(std_val, 1e-8)
                
                final_anomaly_score = min(z_score / 3.0, 1.0)
                is_anomaly = z_score > 3.0
                confidence = 0.6
                model_used = "statistical_fallback"
                explanations = {'z_score': z_score, 'threshold': 3.0}
            else:
                # No detection possible
                final_anomaly_score = 0.0
                is_anomaly = False
                confidence = 0.1
                model_used = "insufficient_data"
                explanations = {'message': 'Insufficient historical data for anomaly detection'}
        
        return AnomalyResult(
            machine_id=machine_id,
            sensor_type=sensor_type,
            timestamp=timestamp,
            value=value,
            anomaly_score=final_anomaly_score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            model_used=model_used,
            explanation=explanations
        )

class PredictiveMaintenancePredictor:
    """Predictive maintenance using advanced ML models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.feature_engineering = {}
        self.model_performance = {}
        
        logger.info("Predictive Maintenance Predictor initialized")
    
    def _engineer_features(self, machine_id: str, sensor_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for predictive maintenance"""
        features = pd.DataFrame()
        
        # Time-based features
        features['hour'] = sensor_data['timestamp'].dt.hour
        features['day_of_week'] = sensor_data['timestamp'].dt.dayofweek
        features['month'] = sensor_data['timestamp'].dt.month
        
        # For each sensor type, create statistical features
        sensor_types = sensor_data['sensor_type'].unique()
        
        for sensor_type in sensor_types:
            sensor_values = sensor_data[sensor_data['sensor_type'] == sensor_type]['value']
            
            if len(sensor_values) > 0:
                # Rolling statistics
                features[f'{sensor_type}_mean_1h'] = sensor_values.rolling('1H').mean()
                features[f'{sensor_type}_std_1h'] = sensor_values.rolling('1H').std()
                features[f'{sensor_type}_max_1h'] = sensor_values.rolling('1H').max()
                features[f'{sensor_type}_min_1h'] = sensor_values.rolling('1H').min()
                
                # Trend features
                features[f'{sensor_type}_trend'] = sensor_values.rolling('2H').apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                
                # Anomaly indicators
                features[f'{sensor_type}_anomaly_count_1h'] = sensor_data[
                    sensor_data['sensor_type'] == sensor_type
                ]['anomaly_score'].rolling('1H').apply(lambda x: (x > 0.7).sum())
        
        return features.fillna(0)
    
    async def predict_maintenance(self, machine_id: str, current_data: pd.DataFrame) -> MaintenancePrediction:
        """Predict maintenance needs for a machine"""
        
        # Engineer features
        features = self._engineer_features(machine_id, current_data)
        
        # For now, implement a rule-based predictor (can be replaced with trained ML model)
        failure_probability = 0.0
        risk_level = 'low'
        recommended_actions = []
        
        # Analyze recent anomalies
        recent_anomalies = current_data[
            current_data['timestamp'] > datetime.utcnow() - timedelta(hours=24)
        ]['anomaly_score']
        
        if len(recent_anomalies) > 0:
            avg_anomaly_score = recent_anomalies.mean()
            anomaly_frequency = (recent_anomalies > 0.5).sum() / len(recent_anomalies)
            
            # Simple risk calculation
            failure_probability = min(0.95, avg_anomaly_score * 0.5 + anomaly_frequency * 0.3)
            
            if failure_probability > 0.8:
                risk_level = 'critical'
                recommended_actions = [
                    'Immediate inspection required',
                    'Consider emergency shutdown',
                    'Contact maintenance team immediately'
                ]
            elif failure_probability > 0.6:
                risk_level = 'high'
                recommended_actions = [
                    'Schedule maintenance within 24 hours',
                    'Increase monitoring frequency',
                    'Prepare replacement parts'
                ]
            elif failure_probability > 0.3:
                risk_level = 'medium'
                recommended_actions = [
                    'Schedule maintenance within 1 week',
                    'Monitor trending parameters',
                    'Review maintenance history'
                ]
            else:
                risk_level = 'low'
                recommended_actions = [
                    'Continue normal operations',
                    'Regular monitoring sufficient'
                ]
        
        # Estimate time to failure
        predicted_failure_time = None
        time_to_failure_hours = None
        
        if failure_probability > 0.3:
            # Simple time estimation based on risk level
            if risk_level == 'critical':
                time_to_failure_hours = 2.0
            elif risk_level == 'high':
                time_to_failure_hours = 24.0
            elif risk_level == 'medium':
                time_to_failure_hours = 168.0  # 1 week
            
            if time_to_failure_hours:
                predicted_failure_time = datetime.utcnow() + timedelta(hours=time_to_failure_hours)
        
        return MaintenancePrediction(
            machine_id=machine_id,
            prediction_timestamp=datetime.utcnow(),
            failure_probability=failure_probability,
            predicted_failure_time=predicted_failure_time,
            time_to_failure_hours=time_to_failure_hours,
            risk_level=risk_level,
            recommended_actions=recommended_actions,
            confidence=0.75,  # Moderate confidence with rule-based approach
            model_version='rule_based_v1.0'
        )

class CognitiveMaintenanceOrchestrator:
    """Orchestrates cognitive maintenance recommendations using reinforcement learning concepts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recommendations_history = defaultdict(list)
        self.action_outcomes = defaultdict(list)
        self.cost_benefit_analysis = {}
        
        logger.info("Cognitive Maintenance Orchestrator initialized")
    
    def _calculate_cost_benefit(self, machine_id: str, prediction: MaintenancePrediction) -> Dict[str, float]:
        """Calculate cost-benefit analysis for maintenance actions"""
        
        # Default cost assumptions (configurable)
        downtime_cost_per_hour = self.config.get('downtime_cost_per_hour', 1000.0)
        maintenance_cost = self.config.get('maintenance_cost', 500.0)
        emergency_repair_multiplier = self.config.get('emergency_repair_multiplier', 3.0)
        
        # Estimate costs
        if prediction.predicted_failure_time:
            # Cost of waiting until failure
            failure_cost = downtime_cost_per_hour * 8 + maintenance_cost * emergency_repair_multiplier
            
            # Cost of preventive maintenance now
            preventive_cost = maintenance_cost
            
            # Savings from preventive action
            potential_savings = failure_cost - preventive_cost
        else:
            potential_savings = 0
            preventive_cost = maintenance_cost
            failure_cost = 0
        
        return {
            'preventive_maintenance_cost': preventive_cost,
            'failure_cost': failure_cost,
            'potential_savings': potential_savings,
            'roi': potential_savings / max(preventive_cost, 1) if preventive_cost > 0 else 0
        }
    
    async def generate_recommendations(self, machine_id: str, anomaly_result: AnomalyResult, 
                                     maintenance_prediction: MaintenancePrediction) -> List[CognitiveRecommendation]:
        """Generate intelligent maintenance recommendations"""
        
        recommendations = []
        
        # Cost-benefit analysis
        cost_benefit = self._calculate_cost_benefit(machine_id, maintenance_prediction)
        
        # Generate recommendations based on risk level and anomalies
        if anomaly_result.is_anomaly and anomaly_result.confidence > 0.7:
            # High-confidence anomaly detected
            if maintenance_prediction.risk_level in ['critical', 'high']:
                rec = CognitiveRecommendation(
                    machine_id=machine_id,
                    recommendation_id=str(uuid.uuid4()),
                    recommendation_type='corrective',
                    description=f"Immediate attention required for {anomaly_result.sensor_type} sensor anomaly. "
                               f"Anomaly score: {anomaly_result.anomaly_score:.2f}",
                    priority=5 if maintenance_prediction.risk_level == 'critical' else 4,
                    estimated_cost_savings=cost_benefit['potential_savings'],
                    implementation_difficulty=3,
                    expected_completion_time=timedelta(hours=2),
                    supporting_evidence=[
                        f"Anomaly detected with {anomaly_result.confidence:.1%} confidence",
                        f"Risk level: {maintenance_prediction.risk_level}",
                        f"Model used: {anomaly_result.model_used}"
                    ],
                    created_at=datetime.utcnow()
                )
                recommendations.append(rec)
        
        # Predictive maintenance recommendations
        if maintenance_prediction.failure_probability > 0.5:
            rec = CognitiveRecommendation(
                machine_id=machine_id,
                recommendation_id=str(uuid.uuid4()),
                recommendation_type='predictive',
                description=f"Predictive maintenance recommended. "
                           f"Failure probability: {maintenance_prediction.failure_probability:.1%}. "
                           f"Estimated time to failure: {maintenance_prediction.time_to_failure_hours:.1f} hours",
                priority=4 if maintenance_prediction.risk_level == 'high' else 3,
                estimated_cost_savings=cost_benefit['potential_savings'],
                implementation_difficulty=2,
                expected_completion_time=timedelta(hours=4),
                supporting_evidence=[
                    f"Failure probability: {maintenance_prediction.failure_probability:.1%}",
                    f"Time to failure: {maintenance_prediction.time_to_failure_hours:.1f} hours",
                    f"ROI: {cost_benefit['roi']:.2f}"
                ],
                created_at=datetime.utcnow()
            )
            recommendations.append(rec)
        
        # Preventive maintenance based on patterns
        if len(self.recommendations_history[machine_id]) > 5:
            # Analyze pattern of past recommendations
            recent_anomalies = sum(1 for rec in self.recommendations_history[machine_id][-10:] 
                                 if rec.recommendation_type == 'corrective')
            
            if recent_anomalies >= 3:  # Pattern of frequent issues
                rec = CognitiveRecommendation(
                    machine_id=machine_id,
                    recommendation_id=str(uuid.uuid4()),
                    recommendation_type='preventive',
                    description=f"Pattern of frequent anomalies detected ({recent_anomalies} in last 10 events). "
                               f"Consider comprehensive maintenance review.",
                    priority=3,
                    estimated_cost_savings=cost_benefit['potential_savings'] * 0.5,
                    implementation_difficulty=4,
                    expected_completion_time=timedelta(hours=8),
                    supporting_evidence=[
                        f"{recent_anomalies} corrective recommendations in recent history",
                        "Pattern suggests underlying mechanical issues",
                        "Preventive overhaul may reduce future incidents"
                    ],
                    created_at=datetime.utcnow()
                )
                recommendations.append(rec)
        
        # Store recommendations for learning
        for rec in recommendations:
            self.recommendations_history[machine_id].append(rec)
        
        return recommendations

class MLPipelineOrchestrator:
    """Main orchestrator for the complete ML pipeline"""
    
    def __init__(self, config_path: str = "config/ml_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.edge_detector = EdgeAnomalyDetector(self.config.get('edge_detection', {}))
        self.maintenance_predictor = PredictiveMaintenancePredictor(self.config.get('prediction', {}))
        self.cognitive_orchestrator = CognitiveMaintenanceOrchestrator(self.config.get('cognitive', {}))
        
        # Database connection
        self.db_engine = None
        self.session_maker = None
        
        # Redis for caching
        self.redis_client = None
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'anomalies_detected': 0,
            'predictions_made': 0,
            'recommendations_generated': 0,
            'processing_time_avg': 0.0
        }
        
        logger.info("ML Pipeline Orchestrator initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load ML pipeline configuration"""
        # Default configuration
        default_config = {
            'edge_detection': {
                'contamination_rate': 0.1,
                'n_estimators': 100,
                'lstm_sequence_length': 20,
                'anomaly_threshold': 0.7
            },
            'prediction': {
                'lookback_hours': 24,
                'prediction_horizon_hours': 48
            },
            'cognitive': {
                'downtime_cost_per_hour': 1000.0,
                'maintenance_cost': 500.0,
                'emergency_repair_multiplier': 3.0
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'pdm_platform',
                'user': 'pdm_user',
                'password': 'pdm_password'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                import yaml
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                
                # Merge with defaults
                for key, value in loaded_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            
            return default_config
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}, using defaults: {e}")
            return default_config
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize database connection
            db_config = self.config['database']
            db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            
            from sqlalchemy import create_engine
            self.db_engine = create_engine(db_url)
            self.session_maker = sessionmaker(bind=self.db_engine)
            
            # Initialize Redis
            redis_config = self.config['redis']
            self.redis_client = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                decode_responses=True
            )
            
            # Test connections
            with self.session_maker() as session:
                session.execute(text("SELECT 1"))
            
            self.redis_client.ping()
            
            logger.info("ML Pipeline components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ML Pipeline: {e}")
            return False
    
    async def process_sensor_data(self, tenant_id: str, machine_id: str, sensor_type: str, 
                                value: float, timestamp: datetime) -> Dict[str, Any]:
        """Process sensor data through the complete ML pipeline"""
        start_time = time.time()
        
        try:
            # Get historical data for context
            historical_data = await self._get_historical_data(tenant_id, machine_id, sensor_type, 
                                                            lookback_hours=24)
            
            # Step 1: Edge anomaly detection
            anomaly_result = await self.edge_detector.detect_anomaly(
                machine_id, sensor_type, value, timestamp, historical_data
            )
            
            # Step 2: Predictive maintenance analysis
            if len(historical_data) > 10:  # Need enough data for prediction
                sensor_df = pd.DataFrame({
                    'timestamp': [timestamp - timedelta(hours=i) for i in range(len(historical_data)-1, -1, -1)],
                    'sensor_type': [sensor_type] * len(historical_data),
                    'value': historical_data,
                    'anomaly_score': [0.0] * len(historical_data)  # Placeholder
                })
                
                maintenance_prediction = await self.maintenance_predictor.predict_maintenance(
                    machine_id, sensor_df
                )
            else:
                # Default prediction for insufficient data
                maintenance_prediction = MaintenancePrediction(
                    machine_id=machine_id,
                    prediction_timestamp=timestamp,
                    failure_probability=0.0,
                    predicted_failure_time=None,
                    time_to_failure_hours=None,
                    risk_level='low',
                    recommended_actions=['Continue normal operations'],
                    confidence=0.1,
                    model_version='insufficient_data'
                )
            
            # Step 3: Cognitive recommendations
            recommendations = await self.cognitive_orchestrator.generate_recommendations(
                machine_id, anomaly_result, maintenance_prediction
            )
            
            # Step 4: Store results
            await self._store_results(tenant_id, anomaly_result, maintenance_prediction, recommendations)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats['total_processed'] += 1
            self.processing_stats['processing_time_avg'] = (
                (self.processing_stats['processing_time_avg'] * (self.processing_stats['total_processed'] - 1) + processing_time) /
                self.processing_stats['total_processed']
            )
            
            if anomaly_result.is_anomaly:
                self.processing_stats['anomalies_detected'] += 1
            
            if maintenance_prediction.failure_probability > 0.3:
                self.processing_stats['predictions_made'] += 1
            
            if recommendations:
                self.processing_stats['recommendations_generated'] += len(recommendations)
            
            # Return comprehensive results
            return {
                'anomaly': asdict(anomaly_result),
                'prediction': asdict(maintenance_prediction),
                'recommendations': [asdict(rec) for rec in recommendations],
                'processing_time_ms': processing_time * 1000,
                'pipeline_version': '2.0'
            }
            
        except Exception as e:
            logger.error(f"Error processing sensor data for {machine_id}.{sensor_type}: {e}")
            return {
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
    
    async def _get_historical_data(self, tenant_id: str, machine_id: str, sensor_type: str, 
                                 lookback_hours: int = 24) -> List[float]:
        """Get historical sensor data from database"""
        try:
            with self.session_maker() as session:
                query = text("""
                    SELECT value FROM sensor_data 
                    WHERE tenant_id = :tenant_id 
                    AND machine_id = :machine_id 
                    AND sensor_type = :sensor_type 
                    AND timestamp > :start_time 
                    ORDER BY timestamp ASC
                """)
                
                start_time = datetime.utcnow() - timedelta(hours=lookback_hours)
                result = session.execute(query, {
                    'tenant_id': tenant_id,
                    'machine_id': machine_id,
                    'sensor_type': sensor_type,
                    'start_time': start_time
                })
                
                return [row.value for row in result]
                
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []
    
    async def _store_results(self, tenant_id: str, anomaly_result: AnomalyResult, 
                           maintenance_prediction: MaintenancePrediction, 
                           recommendations: List[CognitiveRecommendation]):
        """Store ML pipeline results in database"""
        try:
            with self.session_maker() as session:
                # Update sensor_data with anomaly score
                update_query = text("""
                    UPDATE sensor_data 
                    SET anomaly_score = :anomaly_score 
                    WHERE tenant_id = :tenant_id 
                    AND machine_id = :machine_id 
                    AND sensor_type = :sensor_type 
                    AND timestamp = :timestamp
                """)
                
                session.execute(update_query, {
                    'anomaly_score': anomaly_result.anomaly_score,
                    'tenant_id': tenant_id,
                    'machine_id': anomaly_result.machine_id,
                    'sensor_type': anomaly_result.sensor_type,
                    'timestamp': anomaly_result.timestamp
                })
                
                # Store other results in Redis for quick access
                cache_key = f"ml_results:{tenant_id}:{anomaly_result.machine_id}"
                cache_data = {
                    'anomaly': asdict(anomaly_result),
                    'prediction': asdict(maintenance_prediction),
                    'recommendations': [asdict(rec) for rec in recommendations],
                    'updated_at': datetime.utcnow().isoformat()
                }
                
                self.redis_client.setex(
                    cache_key, 
                    3600,  # 1 hour TTL
                    json.dumps(cache_data, default=str)
                )
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to store ML results: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        return {
            **self.processing_stats,
            'edge_detector_models': len(self.edge_detector.models),
            'uptime_seconds': (datetime.utcnow() - datetime.utcnow()).total_seconds(),
            'timestamp': datetime.utcnow().isoformat()
        }

async def main():
    """Main entry point for ML Pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDM Platform v2.0 ML Pipeline')
    parser.add_argument('--config', default='config/ml_config.yaml', help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Run in test mode with sample data')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = MLPipelineOrchestrator(args.config)
    
    if not await orchestrator.initialize():
        logger.error("Failed to initialize ML Pipeline")
        return
    
    if args.test:
        # Test mode with sample data
        logger.info("Running in test mode...")
        
        test_result = await orchestrator.process_sensor_data(
            tenant_id="test-tenant-id",
            machine_id="EG_M001", 
            sensor_type="temperature",
            value=75.5,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Test result: {json.dumps(test_result, indent=2, default=str)}")
        
        # Show performance stats
        stats = orchestrator.get_performance_stats()
        logger.info(f"Performance stats: {json.dumps(stats, indent=2, default=str)}")
    
    else:
        logger.info("ML Pipeline initialized and ready for processing")
        # In production, this would typically run as a service
        # listening for sensor data events
        
        try:
            while True:
                await asyncio.sleep(10)  # Keep running
        except KeyboardInterrupt:
            logger.info("Shutting down ML Pipeline...")

if __name__ == '__main__':
    asyncio.run(main())
