#!/usr/bin/env python3
"""
Simple Pure Dataset Anomaly Detection
Works directly with NYC taxi data without complex preprocessing
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_anomalies_in_pure_data():
    """Detect anomalies in pure NYC taxi data without injection"""
    logger.info("Starting pure data anomaly detection...")
    
    try:
        # Load raw data directly
        data_path = "data/raw/nyc_taxi/fare_per_day.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        raw_data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(raw_data)} records from NYC taxi data")
        
        # Basic data processing
        raw_data['date'] = pd.to_datetime(raw_data['date'])
        raw_data = raw_data.sort_values('date').reset_index(drop=True)
        
        # Use total_fare as the main feature
        if 'total_fare' not in raw_data.columns:
            raise ValueError("Required column 'total_fare' not found in data")
        
        # Prepare data (use recent 70 days if available)
        if len(raw_data) > 70:
            recent_data = raw_data.tail(70).copy()
        else:
            recent_data = raw_data.copy()
        
        # Extract feature values
        fare_data = recent_data['total_fare'].values.reshape(-1, 1)
        logger.info(f"Using {len(fare_data)} data points for analysis")
        logger.info(f"Fare range: ${fare_data.min():.2f} to ${fare_data.max():.2f}")
        
        # Split naturally (80/20)
        split_idx = int(0.8 * len(fare_data))
        train_data = fare_data[:split_idx]
        test_data = fare_data[split_idx:]
        
        logger.info(f"Training set: {len(train_data)} points")
        logger.info(f"Test set: {len(test_data)} points")
        
        # Simple statistical anomaly detection
        train_mean = np.mean(train_data)
        train_std = np.std(train_data)
        
        logger.info(f"Training statistics - Mean: ${train_mean:.2f}, Std: ${train_std:.2f}")
        
        # Detect anomalies using z-score approach
        z_scores = np.abs((test_data - train_mean) / train_std)
        
        # Try different thresholds
        thresholds = [2.0, 2.5, 3.0, 3.5, 4.0]
        results = {}
        
        for threshold in thresholds:
            anomalies = z_scores > threshold
            anomaly_count = np.sum(anomalies)
            anomaly_percentage = (anomaly_count / len(test_data)) * 100
            
            # Quality assessment
            if 0.05 <= anomaly_percentage <= 0.30:  # Reasonable range
                quality_score = 1.0 - abs(anomaly_percentage - 0.15) / 0.15  # Target ~15%
            else:
                quality_score = 0.5 - min(anomaly_percentage, 50) / 100
            
            results[threshold] = {
                'count': int(anomaly_count),
                'percentage': float(anomaly_percentage),
                'quality_score': float(max(0, quality_score)),
                'anomalies': anomalies.flatten().tolist()
            }
            
            logger.info(f"Threshold {threshold}σ: {anomaly_count} anomalies ({anomaly_percentage:.1f}%), Quality: {quality_score:.3f}")
        
        # Find best threshold
        best_threshold = max(results.keys(), key=lambda k: results[k]['quality_score'])
        best_result = results[best_threshold]
        
        logger.info(f"\nBEST RESULT:")
        logger.info(f"  Optimal threshold: {best_threshold}σ")
        logger.info(f"  Anomalies detected: {best_result['count']}/{len(test_data)} ({best_result['percentage']:.1f}%)")
        logger.info(f"  Quality score: {best_result['quality_score']:.3f}")
        
        # Show some examples
        anomaly_indices = np.where(np.array(best_result['anomalies']))[0]
        if len(anomaly_indices) > 0:
            logger.info(f"\nSample anomalies (dates):")
            sample_size = min(5, len(anomaly_indices))
            for i in range(sample_size):
                idx = anomaly_indices[i]
                actual_idx = split_idx + idx
                date = recent_data.iloc[actual_idx]['date']
                fare = recent_data.iloc[actual_idx]['total_fare']
                logger.info(f"  {date.strftime('%Y-%m-%d')}: ${fare:.2f}")
        
        # Save results
        output = {
            'total_data_points': len(fare_data),
            'training_points': len(train_data),
            'test_points': len(test_data),
            'training_mean': float(train_mean),
            'training_std': float(train_std),
            'best_threshold': best_threshold,
            'best_result': best_result,
            'all_results': results
        }
        
        return output
        
    except Exception as e:
        logger.error(f"Pure data analysis failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    results = detect_anomalies_in_pure_data()
    
    if 'error' not in results:
        print("\n=== PURE DATASET ANOMALY DETECTION RESULTS ===")
        print(f"Total data points analyzed: {results['total_data_points']}")
        print(f"Training period: {results['training_points']} points")
        print(f"Detection period: {results['test_points']} points")
        print(f"Training statistics: Mean=${results['training_mean']:.2f}, Std=${results['training_std']:.2f}")
        print(f"\nOPTIMAL DETECTION:")
        print(f"  Threshold: {results['best_threshold']}σ")
        print(f"  Anomalies found: {results['best_result']['count']}")
        print(f"  Anomaly rate: {results['best_result']['percentage']:.1f}%")
        print(f"  Quality score: {results['best_result']['quality_score']:.3f}")
    else:
        print(f"FAILED: {results['error']}")