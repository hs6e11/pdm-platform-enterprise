"""
ML Pipeline Orchestrator v2.0
Cognitive maintenance with edge computing support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import logging
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnomalyPrediction:
    machine_id: str
    timestamp: str
    anomaly_score: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    recommended_actions: List[str]
    confidence: float
    edge_processed: bool = False

class EdgeAnomalyDetector:
    """Lightweight anomaly detector for edge deployment"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            n_estimators=50,
            random_state=42
        )
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> None:
        """Train the edge anomaly detector"""
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        self.is_fitted = True
        logger.info(f"Edge anomaly detector trained on {X.shape[0]} samples")
    
    def predict_anomaly_score(self, X: np.ndarray) -> float:
        """Get anomaly score (0-1, higher = more anomalous)"""
        if not self.is_fitted:
            return 0.0
        
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        score = self.isolation_forest.decision_function(X_scaled)[0]
        # Convert to 0-1 probability
        return max(0, min(1, (1 - score) / 2))

class CognitiveMaintenanceOrchestrator:
    """Main ML orchestrator for predictive maintenance"""
    
    def __init__(self):
        self.edge_detector = EdgeAnomalyDetector()
        self.action_mapping = {
            'temperature': ['Check cooling system', 'Verify thermostat'],
            'vibration': ['Check bearing alignment', 'Lubricate bearings'],
            'pressure': ['Inspect seals', 'Check fluid levels']
        }
    
    async def process_real_time_data(self, sensor_data: Dict[str, Any]) -> AnomalyPrediction:
        """Process real-time sensor data"""
        try:
            # Extract numerical features
            features = self._extract_features(sensor_data)
            
            # Anomaly detection
            anomaly_score = self.edge_detector.predict_anomaly_score(features)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(sensor_data, anomaly_score)
            
            # Classify severity
            severity = self._classify_severity(anomaly_score)
            
            prediction = AnomalyPrediction(
                machine_id=sensor_data.get('machine_id', 'unknown'),
                timestamp=datetime.utcnow().isoformat(),
                anomaly_score=anomaly_score,
                severity=severity,
                recommended_actions=recommendations,
                confidence=0.85,
                edge_processed=True
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"ML processing error: {e}")
            return self._create_error_prediction(sensor_data)
    
    def _extract_features(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from sensor data"""
        sensors = sensor_data.get('sensors', {})
        features = [
            sensors.get('temperature', 0.0),
            sensors.get('pressure', 0.0),
            sensors.get('vibration', 0.0),
            sensors.get('power_consumption', 0.0)
        ]
        return np.array(features, dtype=float)
    
    def _generate_recommendations(self, sensor_data: Dict[str, Any], 
                                 anomaly_score: float) -> List[str]:
        """Generate maintenance recommendations"""
        recommendations = []
        sensors = sensor_data.get('sensors', {})
        
        # Rule-based recommendations
        if sensors.get('temperature', 0) > 80:
            recommendations.extend(self.action_mapping['temperature'])
        
        if sensors.get('vibration', 0) > 30:
            recommendations.extend(self.action_mapping['vibration'])
        
        if anomaly_score > 0.7:
            recommendations.append('Schedule immediate inspection')
        
        return recommendations
    
    def _classify_severity(self, anomaly_score: float) -> str:
        """Classify severity based on anomaly score"""
        if anomaly_score > 0.8:
            return 'critical'
        elif anomaly_score > 0.6:
            return 'high'
        elif anomaly_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _create_error_prediction(self, sensor_data: Dict[str, Any]) -> AnomalyPrediction:
        """Create error prediction when processing fails"""
        return AnomalyPrediction(
            machine_id=sensor_data.get('machine_id', 'unknown'),
            timestamp=datetime.utcnow().isoformat(),
            anomaly_score=0.0,
            severity='low',
            recommended_actions=['Check sensor connectivity'],
            confidence=0.0,
            edge_processed=False
        )
    
    async def train_models(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Train models using historical data"""
        try:
            feature_columns = ['temperature', 'pressure', 'vibration', 'power_consumption']
            X = historical_data[feature_columns].fillna(0).values
            
            self.edge_detector.fit(X)
            
            return {
                'edge_detector_trained': True,
                'samples_used': len(X),
                'feature_columns': feature_columns
            }
        
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return {'error': str(e)}

# Example usage
async def main():
    orchestrator = CognitiveMaintenanceOrchestrator()
    
    # Simulate sensor data
    sample_data = {
        'machine_id': 'EG_M001',
        'sensors': {
            'temperature': 75.5,
            'pressure': 2.8,
            'vibration': 25.3,
            'power_consumption': 15.2
        }
    }
    
    prediction = await orchestrator.process_real_time_data(sample_data)
    print(f"Anomaly Score: {prediction.anomaly_score:.3f}")
    print(f"Severity: {prediction.severity}")
    print(f"Recommendations: {prediction.recommended_actions}")

if __name__ == "__main__":
    asyncio.run(main())
