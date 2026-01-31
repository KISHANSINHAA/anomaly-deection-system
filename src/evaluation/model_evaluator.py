"""
Production Evaluation Framework for Anomaly Detection
Comprehensive evaluation with constraint checking for production deployment
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix
)
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ProductionConstraints:
    """Production deployment constraints for anomaly detection models"""
    min_recall: float = 0.85        # Minimum required recall
    min_f1_score: float = 0.75      # Minimum required F1-score
    target_precision: float = 0.90  # Target precision (optional)
    max_false_discovery_rate: float = 0.15  # Max false discovery rate (1 - precision)


class ProductionModelEvaluator:
    """Evaluator for production-ready anomaly detection models"""
    
    def __init__(self, constraints: Optional[ProductionConstraints] = None):
        """
        Initialize production evaluator
        
        Args:
            constraints: Production constraints for model validation
        """
        self.constraints = constraints or ProductionConstraints()
        self.evaluation_history = []
    
    def evaluate_model(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray, 
                      scores: Optional[np.ndarray] = None,
                      model_name: str = "Model",
                      timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model for production readiness
        
        Args:
            y_true: True binary labels (1=anomaly, 0=normal)
            y_pred: Predicted binary labels
            scores: Anomaly scores (optional, for detailed analysis)
            model_name: Name of the model being evaluated
            timestamp: Evaluation timestamp
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate basic metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_discovery_rate = fp / (fp + tp) if (fp + tp) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Business metrics
        business_metrics = self._calculate_business_impact(tp, fp, fn, tn)
        
        # Constraint validation
        constraint_results = self._validate_constraints(precision, recall, f1)
        
        # Detailed analysis if scores provided
        score_analysis = {}
        if scores is not None:
            score_analysis = self._analyze_scores(scores, y_true, y_pred)
        
        # Performance summary
        results = {
            'model_name': model_name,
            'timestamp': timestamp.isoformat(),
            'basic_metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'specificity': specificity,
                'false_discovery_rate': false_discovery_rate,
                'false_negative_rate': false_negative_rate
            },
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'matrix': cm.tolist()
            },
            'business_metrics': business_metrics,
            'constraint_validation': constraint_results,
            'score_analysis': score_analysis,
            'production_ready': constraint_results['overall_pass'],
            'recommendations': self._generate_recommendations(
                precision, recall, f1, constraint_results['overall_pass']
            )
        }
        
        # Store evaluation history
        self.evaluation_history.append(results)
        
        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        logger.info(f"Production ready: {constraint_results['overall_pass']}")
        
        return results
    
    def _calculate_business_impact(self, tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
        """
        Calculate business impact metrics
        
        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
            tn: True negatives
            
        Returns:
            Dictionary with business impact metrics
        """
        # Business cost assumptions (can be customized)
        cost_per_false_positive = 100      # Cost of false alarm
        cost_per_false_negative = 1000     # Cost of missed anomaly
        value_per_true_positive = 500      # Value of catching real anomaly
        cost_per_true_negative = 10        # Operational cost savings
        
        total_cost = (
            fp * cost_per_false_positive +
            fn * cost_per_false_negative -
            tp * value_per_true_positive -
            tn * cost_per_true_negative
        )
        
        total_predictions = tp + fp + fn + tn
        
        return {
            'total_business_cost': float(total_cost),
            'cost_per_prediction': float(total_cost / total_predictions) if total_predictions > 0 else 0,
            'roi_from_detections': float(tp * value_per_true_positive),
            'cost_of_false_alarms': float(fp * cost_per_false_positive),
            'cost_of_missed_anomalies': float(fn * cost_per_false_negative),
            'net_business_value': float(
                tp * value_per_true_positive - 
                fp * cost_per_false_positive - 
                fn * cost_per_false_negative
            )
        }
    
    def _validate_constraints(self, precision: float, recall: float, f1: float) -> Dict[str, Any]:
        """
        Validate model against production constraints
        
        Args:
            precision: Model precision
            recall: Model recall
            f1: Model F1-score
            
        Returns:
            Dictionary with constraint validation results
        """
        meets_min_recall = recall >= self.constraints.min_recall
        meets_min_f1 = f1 >= self.constraints.min_f1_score
        meets_target_precision = precision >= self.constraints.target_precision
        meets_max_fdr = (1 - precision) <= self.constraints.max_false_discovery_rate
        
        overall_pass = (
            meets_min_recall and 
            meets_min_f1 and 
            meets_max_fdr
        )
        
        return {
            'meets_min_recall': meets_min_recall,
            'meets_min_f1_score': meets_min_f1,
            'meets_target_precision': meets_target_precision,
            'meets_max_false_discovery_rate': meets_max_fdr,
            'overall_pass': overall_pass,
            'required_improvements': self._get_required_improvements(
                precision, recall, f1
            )
        }
    
    def _get_required_improvements(self, precision: float, recall: float, f1: float) -> List[str]:
        """
        Get list of required improvements for constraint satisfaction
        
        Args:
            precision: Current precision
            recall: Current recall
            f1: Current F1-score
            
        Returns:
            List of improvement recommendations
        """
        improvements = []
        
        if recall < self.constraints.min_recall:
            gap = self.constraints.min_recall - recall
            improvements.append(f"Increase recall by {gap:.3f} (current: {recall:.3f})")
            
        if f1 < self.constraints.min_f1_score:
            gap = self.constraints.min_f1_score - f1
            improvements.append(f"Increase F1-score by {gap:.3f} (current: {f1:.3f})")
            
        if precision < self.constraints.target_precision:
            gap = self.constraints.target_precision - precision
            improvements.append(f"Increase precision by {gap:.3f} (current: {precision:.3f})")
            
        if (1 - precision) > self.constraints.max_false_discovery_rate:
            gap = (1 - precision) - self.constraints.max_false_discovery_rate
            improvements.append(f"Reduce false discovery rate by {gap:.3f} (current: {1-precision:.3f})")
            
        return improvements
    
    def _analyze_scores(self, scores: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Detailed analysis of anomaly scores
        
        Args:
            scores: Anomaly scores
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with score analysis
        """
        # Score distribution
        score_analysis = {
            'score_distribution': {
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'median': float(np.median(scores))
            },
            'score_by_class': {
                'normal_scores': scores[y_true == 0].tolist(),
                'anomaly_scores': scores[y_true == 1].tolist()
            }
        }
        
        # PR curve analysis
        try:
            precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, scores)
            auc_pr = average_precision_score(y_true, scores)
            
            # Find optimal threshold
            f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
            f1_scores = np.nan_to_num(f1_scores)
            optimal_idx = np.argmax(f1_scores)
            
            score_analysis['pr_curve'] = {
                'auc_pr': float(auc_pr),
                'optimal_f1': float(f1_scores[optimal_idx]),
                'optimal_precision': float(precision_vals[optimal_idx]),
                'optimal_recall': float(recall_vals[optimal_idx]),
                'optimal_threshold': float(thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1])
            }
        except Exception as e:
            logger.warning(f"Could not compute PR curve: {e}")
            score_analysis['pr_curve'] = {}
        
        return score_analysis
    
    def _generate_recommendations(self, precision: float, recall: float, f1: float, 
                                meets_constraints: bool) -> List[str]:
        """
        Generate actionable recommendations
        
        Args:
            precision: Model precision
            recall: Model recall
            f1: Model F1-score
            meets_constraints: Whether constraints are met
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if meets_constraints:
            recommendations.append("✅ Model meets all production requirements")
            recommendations.append("✅ Ready for deployment")
            recommendations.append("✅ Consider A/B testing in production environment")
        else:
            recommendations.append("❌ Model does not meet production requirements")
            recommendations.append("🔧 Required improvements:")
            if recall < self.constraints.min_recall:
                recommendations.append(f"   - Improve recall (current: {recall:.3f}, required: {self.constraints.min_recall})")
            if f1 < self.constraints.min_f1_score:
                recommendations.append(f"   - Improve F1-score (current: {f1:.3f}, required: {self.constraints.min_f1_score})")
        
        # General performance recommendations
        if precision > 0.95 and recall < 0.7:
            recommendations.append("⚠️  Very high precision but low recall - may be overly conservative")
        elif recall > 0.95 and precision < 0.7:
            recommendations.append("⚠️  Very high recall but low precision - may generate too many false alarms")
        
        if f1 < 0.6:
            recommendations.append("🔧 F1-score very low - consider retraining with different parameters")
        
        return recommendations
    
    def compare_models(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models based on production criteria
        
        Args:
            evaluation_results: List of evaluation results from different models
            
        Returns:
            Dictionary with comparison analysis
        """
        if not evaluation_results:
            return {}
        
        # Extract key metrics
        model_names = [result['model_name'] for result in evaluation_results]
        precisions = [result['basic_metrics']['precision'] for result in evaluation_results]
        recalls = [result['basic_metrics']['recall'] for result in evaluation_results]
        f1_scores = [result['basic_metrics']['f1_score'] for result in evaluation_results]
        production_ready = [result['production_ready'] for result in evaluation_results]
        
        # Find best performers
        best_precision_idx = np.argmax(precisions)
        best_recall_idx = np.argmax(recalls)
        best_f1_idx = np.argmax(f1_scores)
        ready_count = sum(production_ready)
        
        comparison = {
            'summary': {
                'total_models': len(evaluation_results),
                'production_ready_models': ready_count,
                'best_precision_model': model_names[best_precision_idx],
                'best_recall_model': model_names[best_recall_idx],
                'best_f1_model': model_names[best_f1_idx],
                'production_ready_model_names': [name for name, ready in zip(model_names, production_ready) if ready]
            },
            'detailed_metrics': {
                'model_names': model_names,
                'precisions': precisions,
                'recalls': recalls,
                'f1_scores': f1_scores,
                'production_ready': production_ready
            },
            'recommendations': self._generate_comparison_recommendations(
                evaluation_results, production_ready
            )
        }
        
        return comparison
    
    def _generate_comparison_recommendations(self, 
                                           evaluation_results: List[Dict[str, Any]],
                                           production_ready: List[bool]) -> List[str]:
        """
        Generate recommendations from model comparison
        
        Args:
            evaluation_results: List of evaluation results
            production_ready: List of production readiness flags
            
        Returns:
            List of recommendations
        """
        recommendations = []
        ready_count = sum(production_ready)
        total_models = len(evaluation_results)
        
        if ready_count == total_models:
            recommendations.append("✅ All models meet production requirements")
            recommendations.append("🏆 Choose based on additional criteria (business impact, latency, etc.)")
        elif ready_count > 0:
            recommendations.append(f"✅ {ready_count}/{total_models} models are production-ready")
            recommendations.append("🔧 Deploy production-ready models, improve others")
        else:
            recommendations.append("❌ No models meet production requirements")
            recommendations.append("🔧 Significant model improvement needed")
        
        # Performance spread analysis
        precisions = [r['basic_metrics']['precision'] for r in evaluation_results]
        recalls = [r['basic_metrics']['recall'] for r in evaluation_results]
        
        precision_spread = np.max(precisions) - np.min(precisions)
        recall_spread = np.max(recalls) - np.min(recalls)
        
        if precision_spread > 0.3:
            recommendations.append("📊 Large precision variation - investigate model differences")
        if recall_spread > 0.3:
            recommendations.append("📊 Large recall variation - check data preprocessing consistency")
        
        return recommendations


def quick_production_check(y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          scores: Optional[np.ndarray] = None,
                          constraints: Optional[ProductionConstraints] = None) -> Dict[str, Any]:
    """
    Quick production readiness check
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        scores: Anomaly scores (optional)
        constraints: Production constraints
        
    Returns:
        Dictionary with quick evaluation results
    """
    evaluator = ProductionModelEvaluator(constraints)
    results = evaluator.evaluate_model(y_true, y_pred, scores, "Quick Check")
    
    return {
        'precision': results['basic_metrics']['precision'],
        'recall': results['basic_metrics']['recall'],
        'f1_score': results['basic_metrics']['f1_score'],
        'production_ready': results['production_ready'],
        'constraint_violations': results['constraint_validation']['required_improvements'],
        'top_recommendations': results['recommendations'][:3]
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate predictions with different performance levels
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    # Good model (meets constraints)
    good_scores = np.random.beta(2, 5, n_samples)
    good_scores[y_true == 1] = np.random.beta(5, 2, np.sum(y_true == 1))
    good_pred = (good_scores > np.percentile(good_scores, 90)).astype(int)
    
    # Poor model (doesn't meet constraints)
    poor_scores = np.random.uniform(0, 1, n_samples)
    poor_pred = (poor_scores > 0.5).astype(int)
    
    try:
        # Test production evaluator
        evaluator = ProductionModelEvaluator()
        
        # Evaluate good model
        good_results = evaluator.evaluate_model(y_true, good_pred, good_scores, "Good Model")
        print("Good Model Results:")
        print(f"  Precision: {good_results['basic_metrics']['precision']:.3f}")
        print(f"  Recall: {good_results['basic_metrics']['recall']:.3f}")
        print(f"  F1-Score: {good_results['basic_metrics']['f1_score']:.3f}")
        print(f"  Production Ready: {good_results['production_ready']}")
        
        # Evaluate poor model
        poor_results = evaluator.evaluate_model(y_true, poor_pred, poor_scores, "Poor Model")
        print("\nPoor Model Results:")
        print(f"  Precision: {poor_results['basic_metrics']['precision']:.3f}")
        print(f"  Recall: {poor_results['basic_metrics']['recall']:.3f}")
        print(f"  F1-Score: {poor_results['basic_metrics']['f1_score']:.3f}")
        print(f"  Production Ready: {poor_results['production_ready']}")
        
        # Compare models
        comparison = evaluator.compare_models([good_results, poor_results])
        print(f"\nComparison: {comparison['summary']['production_ready_models']}/2 models ready")
        
    except Exception as e:
        print(f"Error: {e}")