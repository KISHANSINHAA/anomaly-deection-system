"""
Precision-Focused Dynamic Thresholding for Anomaly Detection
Adaptive percentile selection to maximize precision while maintaining recall constraints
"""

import numpy as np
from typing import Tuple, List, Dict
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class PrecisionFocusedThresholding:
    """Dynamic thresholding optimized for precision maximization"""
    
    def __init__(self, 
                 min_recall: float = 0.85,
                 target_precision: float = 0.90,
                 percentile_range: Tuple[float, float] = (95.0, 99.5),
                 step_size: float = 0.5):
        """
        Initialize precision-focused thresholding
        
        Args:
            min_recall: Minimum acceptable recall (default: 0.85)
            target_precision: Target precision to achieve (default: 0.90)
            percentile_range: Range of percentiles to test (default: 95.0 to 99.5)
            step_size: Step size for percentile testing (default: 0.5)
        """
        self.min_recall = min_recall
        self.target_precision = target_precision
        self.percentile_range = percentile_range
        self.step_size = step_size
        self.optimal_threshold = None
        self.optimal_percentile = None
        self.performance_history = []
        
    def find_optimal_threshold(self, 
                             scores: np.ndarray, 
                             ground_truth: np.ndarray) -> Tuple[float, Dict]:
        """
        Find optimal threshold that maximizes precision while maintaining minimum recall
        
        Args:
            scores: Anomaly scores from model predictions
            ground_truth: True anomaly labels
            
        Returns:
            Tuple of (optimal_threshold, performance_metrics)
        """
        logger.info(f"Finding optimal threshold for {len(scores)} samples")
        logger.info(f"Constraints: min_recall={self.min_recall}, target_precision={self.target_precision}")
        
        # Test different percentiles
        percentiles = np.arange(
            self.percentile_range[0], 
            self.percentile_range[1] + self.step_size, 
            self.step_size
        )
        
        best_precision = 0
        best_metrics = {}
        best_threshold = None
        best_percentile = None
        
        for percentile in percentiles:
            threshold = np.percentile(scores, percentile)
            predictions = (scores > threshold).astype(int)
            
            # Calculate metrics
            tp = np.sum((predictions == 1) & (ground_truth == 1))
            fp = np.sum((predictions == 1) & (ground_truth == 0))
            fn = np.sum((predictions == 0) & (ground_truth == 1))
            tn = np.sum((predictions == 0) & (ground_truth == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(ground_truth)
            
            # Store performance history
            self.performance_history.append({
                'percentile': percentile,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })
            
            # Check if meets minimum recall requirement
            if recall >= self.min_recall:
                # Prefer higher precision within recall constraint
                if precision > best_precision:
                    best_precision = precision
                    best_threshold = threshold
                    best_percentile = percentile
                    best_metrics = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'accuracy': accuracy,
                        'tp': tp,
                        'fp': fp,
                        'fn': fn,
                        'tn': tn,
                        'percentile': percentile,
                        'threshold': threshold
                    }
            
            logger.debug(f"Perc: {percentile:.1f}, Thresh: {threshold:.6f}, "
                        f"P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")
        
        # Set optimal parameters
        self.optimal_threshold = best_threshold
        self.optimal_percentile = best_percentile
        
        logger.info(f"Optimal threshold found:")
        logger.info(f"  Percentile: {best_percentile:.1f}")
        logger.info(f"  Threshold: {best_threshold:.6f}")
        logger.info(f"  Precision: {best_metrics['precision']:.3f}")
        logger.info(f"  Recall: {best_metrics['recall']:.3f}")
        logger.info(f"  F1-Score: {best_metrics['f1']:.3f}")
        
        return best_threshold, best_metrics
    
    def predict_with_threshold(self, scores: np.ndarray) -> np.ndarray:
        """Predict anomalies using the optimal threshold"""
        if self.optimal_threshold is None:
            raise ValueError("Must call find_optimal_threshold first")
        
        return (scores > self.optimal_threshold).astype(int)
    
    def get_performance_summary(self) -> Dict:
        """Get summary of threshold optimization results"""
        if not self.performance_history:
            return {}
        
        # Find best performing configurations
        valid_configs = [p for p in self.performance_history if p['recall'] >= self.min_recall]
        
        if not valid_configs:
            return {'error': 'No configurations meet minimum recall requirement'}
        
        # Best precision configuration
        best_prec_config = max(valid_configs, key=lambda x: x['precision'])
        
        # Best F1 configuration
        best_f1_config = max(valid_configs, key=lambda x: x['f1'])
        
        return {
            'constraint_satisfied': len(valid_configs) > 0,
            'total_configurations_tested': len(self.performance_history),
            'valid_configurations': len(valid_configs),
            'best_precision_config': {
                'percentile': best_prec_config['percentile'],
                'threshold': best_prec_config['threshold'],
                'precision': best_prec_config['precision'],
                'recall': best_prec_config['recall'],
                'f1': best_prec_config['f1']
            },
            'best_f1_config': {
                'percentile': best_f1_config['percentile'],
                'threshold': best_f1_config['threshold'],
                'precision': best_f1_config['precision'],
                'recall': best_f1_config['recall'],
                'f1': best_f1_config['f1']
            },
            'optimal_config': {
                'percentile': self.optimal_percentile,
                'threshold': self.optimal_threshold,
                'metrics': self.performance_history[-1] if self.performance_history else {}
            }
        }

class AdaptivePercentileSelector:
    """Adaptive percentile selection based on data characteristics"""
    
    def __init__(self, 
                 base_percentile: float = 97.0,
                 adaptation_factor: float = 0.1):
        """
        Initialize adaptive percentile selector
        
        Args:
            base_percentile: Starting percentile (default: 97.0)
            adaptation_factor: Factor for adapting percentile based on data (default: 0.1)
        """
        self.base_percentile = base_percentile
        self.adaptation_factor = adaptation_factor
        self.learned_percentiles = []
        
    def adapt_percentile(self, scores: np.ndarray, recent_performance: Dict = None) -> float:
        """
        Adapt percentile based on score distribution and recent performance
        
        Args:
            scores: Current anomaly scores
            recent_performance: Recent performance metrics
            
        Returns:
            Adapted percentile value
        """
        # Start with base percentile
        adapted_percentile = self.base_percentile
        
        # Adapt based on score distribution
        score_std = np.std(scores)
        score_skew = stats.skew(scores)
        
        # If scores are highly skewed, adjust percentile
        if abs(score_skew) > 1.0:
            skew_adjustment = self.adaptation_factor * score_skew * 10
            adapted_percentile += skew_adjustment
            logger.debug(f"Skew adjustment: {skew_adjustment:.2f}")
        
        # Adapt based on recent performance
        if recent_performance:
            current_precision = recent_performance.get('precision', 0)
            current_recall = recent_performance.get('recall', 0)
            
            # If precision is too low, increase percentile
            if current_precision < 0.8:
                adapted_percentile += 2.0
                logger.debug("Precision low, increasing percentile")
            
            # If recall is too high (too many false negatives), decrease percentile
            elif current_recall > 0.95:
                adapted_percentile -= 1.0
                logger.debug("Recall high, decreasing percentile")
        
        # Ensure percentile stays within reasonable bounds
        adapted_percentile = np.clip(adapted_percentile, 90.0, 99.9)
        
        self.learned_percentiles.append(adapted_percentile)
        logger.debug(f"Adapted percentile: {adapted_percentile:.2f}")
        
        return adapted_percentile

def precision_threshold_analysis(scores: np.ndarray, 
                               ground_truth: np.ndarray) -> Dict:
    """
    Comprehensive analysis of precision-focused thresholding
    
    Args:
        scores: Anomaly scores
        ground_truth: True labels
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    # Initialize thresholding components
    threshold_optimizer = PrecisionFocusedThresholding(
        min_recall=0.85,
        target_precision=0.90
    )
    
    adaptive_selector = AdaptivePercentileSelector(base_percentile=97.0)
    
    # Find optimal threshold
    optimal_threshold, metrics = threshold_optimizer.find_optimal_threshold(scores, ground_truth)
    
    # Test adaptive percentile selection
    adaptive_percentile = adaptive_selector.adapt_percentile(scores, metrics)
    adaptive_threshold = np.percentile(scores, adaptive_percentile)
    adaptive_predictions = (scores > adaptive_threshold).astype(int)
    
    # Calculate adaptive performance
    tp_adaptive = np.sum((adaptive_predictions == 1) & (ground_truth == 1))
    fp_adaptive = np.sum((adaptive_predictions == 1) & (ground_truth == 0))
    fn_adaptive = np.sum((adaptive_predictions == 0) & (ground_truth == 1))
    
    adaptive_precision = tp_adaptive / (tp_adaptive + fp_adaptive) if (tp_adaptive + fp_adaptive) > 0 else 0
    adaptive_recall = tp_adaptive / (tp_adaptive + fn_adaptive) if (tp_adaptive + fn_adaptive) > 0 else 0
    
    # Generate performance report
    summary = threshold_optimizer.get_performance_summary()
    
    analysis_results = {
        'optimal_threshold': {
            'value': optimal_threshold,
            'percentile': threshold_optimizer.optimal_percentile,
            'metrics': metrics
        },
        'adaptive_threshold': {
            'value': adaptive_threshold,
            'percentile': adaptive_percentile,
            'precision': adaptive_precision,
            'recall': adaptive_recall
        },
        'analysis_summary': summary,
        'score_statistics': {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores),
            'skewness': stats.skew(scores)
        },
        'recommendations': []
    }
    
    # Generate recommendations
    if metrics['precision'] >= 0.90:
        analysis_results['recommendations'].append("Excellent precision achieved - model ready for deployment")
    elif metrics['precision'] >= 0.80:
        analysis_results['recommendations'].append("Good precision - consider minor threshold adjustments")
    else:
        analysis_results['recommendations'].append("Precision below target - review model or data quality")
    
    if metrics['recall'] < 0.85:
        analysis_results['recommendations'].append("Recall constraint violated - adjust threshold strategy")
    
    logger.info("Precision threshold analysis completed")
    return analysis_results

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Normal scores (majority)
    normal_scores = np.random.normal(0.1, 0.05, int(n_samples * 0.95))
    
    # Anomalous scores (minority)
    anomalous_scores = np.random.normal(0.8, 0.1, int(n_samples * 0.05))
    
    scores = np.concatenate([normal_scores, anomalous_scores])
    ground_truth = np.concatenate([np.zeros(int(n_samples * 0.95)), np.ones(int(n_samples * 0.05))])
    
    # Shuffle data
    indices = np.random.permutation(len(scores))
    scores = scores[indices]
    ground_truth = ground_truth[indices]
    
    # Run analysis
    results = precision_threshold_analysis(scores, ground_truth)
    
    print("Precision Threshold Analysis Results:")
    print(f"Optimal Threshold: {results['optimal_threshold']['value']:.6f}")
    print(f"Precision: {results['optimal_threshold']['metrics']['precision']:.3f}")
    print(f"Recall: {results['optimal_threshold']['metrics']['recall']:.3f}")
    print(f"F1-Score: {results['optimal_threshold']['metrics']['f1']:.3f}")