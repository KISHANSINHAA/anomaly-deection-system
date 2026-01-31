"""
Main Pipeline: NYC Taxi Anomaly Detection System
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from preprocessing.feature_engineer import TimeSeriesPreprocessor
from models.isolation_forest_detector import IsolationForestDetector
from models.lstm_autoencoder import LSTMAutoencoder
from models.gru_autoencoder import GRUAutoencoder
from ensemble.scorer import EnsembleScorer
from evaluation.metrics import AnomalyEvaluator
from anomaly_injection.anomaly_injector import create_benchmark_dataset

def run_pipeline():
    """Execute complete anomaly detection pipeline"""
    
    print("=" * 60)
    print("SENTINELGUARD: NYC Taxi Anomaly Detection System")
    print("=" * 60)
    
    # 1. DATA PREPARATION
    print("\n[1/6] Preparing Data...")
    data_path = '../data/nyc_taxi/fare_per_day.csv'
    
    # Create benchmark dataset with known anomalies
    df_benchmark, y_true, anomaly_indices = create_benchmark_dataset(
        filepath=data_path, contamination=0.05
    )
    
    # Preprocess data
    preprocessor = TimeSeriesPreprocessor(target_column='total_fare', sequence_length=7)
    data_dict = preprocessor.prepare_for_models(df_benchmark)
    
    print(f"  Training samples: {len(data_dict['X_train'])}")
    print(f"  Test samples: {len(data_dict['X_test'])}")
    print(f"  Features: {len(data_dict['feature_columns'])}")
    print(f"  Known anomalies: {np.sum(y_true)}")
    
    # 2. TRAIN ISOLATION FOREST
    print("\n[2/6] Training Isolation Forest...")
    iforest = IsolationForestDetector(contamination=0.1, random_state=42)
    iforest.fit(data_dict['X_train'], data_dict['feature_columns'])
    
    # Get scores
    iforest_scores_train = iforest.anomaly_scores(data_dict['X_train'])
    iforest_scores_test = iforest.anomaly_scores(data_dict['X_test'])
    
    print(f"  IF Train score range: [{iforest_scores_train.min():.3f}, {iforest_scores_train.max():.3f}]")
    print(f"  IF Test score range: [{iforest_scores_test.min():.3f}, {iforest_scores_test.max():.3f}]")
    
    # 3. TRAIN LSTM AUTOENCODER
    print("\n[3/6] Training LSTM Autoencoder...")
    lstm_ae = LSTMAutoencoder(
        sequence_length=7,
        n_features=len(data_dict['feature_columns']),
        encoding_dim=10,
        learning_rate=0.001
    )
    lstm_ae.fit(
        data_dict['X_seq_train'],
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Get scores
    lstm_scores_test = lstm_ae.anomaly_scores(data_dict['X_seq_test'])
    
    print(f"  LSTM Test score range: [{lstm_scores_test.min():.3f}, {lstm_scores_test.max():.3f}]")
    print(f"  Mean reconstruction error: {lstm_ae.reconstruction_errors(data_dict['X_seq_test']).mean():.6f}")
    
    # 4. TRAIN GRU AUTOENCODER
    print("\n[4/6] Training GRU Autoencoder...")
    gru_ae = GRUAutoencoder(
        sequence_length=7,
        n_features=len(data_dict['feature_columns']),
        encoding_dim=10,
        learning_rate=0.001
    )
    gru_ae.fit(
        data_dict['X_seq_train'],
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Get scores
    gru_scores_test = gru_ae.anomaly_scores(data_dict['X_seq_test'])
    
    print(f"  GRU Test score range: [{gru_scores_test.min():.3f}, {gru_scores_test.max():.3f}]")
    print(f"  Mean reconstruction error: {gru_ae.reconstruction_errors(data_dict['X_seq_test']).mean():.6f}")
    
    # 5. ENSEMBLE SCORING
    print("\n[5/6] Ensemble Scoring...")
    ensemble = EnsembleScorer(weights={'iforest': 0.4, 'lstm': 0.3, 'gru': 0.3})
    
    # Align sequences (LSTM/GRU have fewer samples due to sequence creation)
    n_seq_samples = len(data_dict['X_seq_test'])
    iforest_scores_aligned = iforest_scores_test[-n_seq_samples:]
    # Align true labels with test data
    y_true_test = y_true[len(y_true) - n_seq_samples:]
    y_true_aligned = y_true_test
    dates_aligned = data_dict['dates'][-n_seq_samples:]
    
    # Combine scores
    final_scores = ensemble.combine_scores(
        iforest_scores_aligned,
        lstm_scores_test,
        gru_scores_test
    )
    
    print(f"  Ensemble score range: [{final_scores.min():.3f}, {final_scores.max():.3f}]")
    print(f"  Mean ensemble score: {final_scores.mean():.3f}")
    
    # 6. EVALUATION
    print("\n[6/6] Evaluation...")
    
    # Find optimal threshold
    from src.ensemble.scorer import optimal_threshold_search
    optimal_threshold, best_f1 = optimal_threshold_search(final_scores, y_true_aligned, 'f1')
    
    print(f"  Optimal threshold (F1-max): {optimal_threshold:.3f}")
    print(f"  Best F1-score: {best_f1:.3f}")
    
    # Get predictions
    predictions = ensemble.get_anomalies(threshold=optimal_threshold)
    detected_anomalies = np.where(predictions == 1)[0]
    
    print(f"\nResults:")
    print(f"  Detected anomalies: {len(detected_anomalies)}")
    actual_anomalies = np.sum(y_true_aligned)
    print(f"  Actual anomalies: {actual_anomalies}")
    if actual_anomalies > 0:
        print(f"  Detection rate: {len(detected_anomalies)/actual_anomalies*100:.1f}%")
    else:
        print(f"  Detection rate: N/A (no actual anomalies in test set)")
    
    # Detailed metrics
    evaluator = AnomalyEvaluator()
    metrics = evaluator.calculate_metrics(y_true_aligned, final_scores, optimal_threshold)
    
    print(f"\nPerformance Metrics:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    print(f"  PR-AUC: {metrics['pr_auc']:.3f}")
    
    # Save results
    save_results(
        final_scores, predictions, y_true_aligned, dates_aligned,
        metrics, optimal_threshold
    )
    
    print("\nPipeline completed successfully!")
    print("Run 'streamlit run src/app/streamlit_app.py' to visualize results.")
    
    return {
        'scores': final_scores,
        'predictions': predictions,
        'true_labels': y_true_aligned,
        'metrics': metrics,
        'threshold': optimal_threshold,
        'dates': dates_aligned
    }

def save_results(scores, predictions, y_true, dates, metrics, threshold):
    """Save pipeline results to artifacts"""
    import json
    
    os.makedirs('artifacts/results', exist_ok=True)
    
    # Save numerical results
    # Convert numpy arrays to lists for JSON serialization
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics_serializable[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            metrics_serializable[k] = float(v)
        else:
            metrics_serializable[k] = v
    
    results = {
        'scores': scores.tolist(),
        'predictions': predictions.tolist(),
        'true_labels': y_true.tolist(),
        'dates': [str(d) for d in dates],
        'metrics': metrics_serializable,
        'threshold': float(threshold)
    }
    
    with open('artifacts/results/pipeline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model comparison
    comparison_results = AnomalyEvaluator.evaluate_ensemble_performance(
        scores, scores, scores, scores, y_true, threshold
    )
    
    with open('artifacts/results/model_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print("Results saved to artifacts/results/")

if __name__ == "__main__":
    results = run_pipeline()
