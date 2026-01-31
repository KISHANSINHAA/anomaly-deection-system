"""
Test suite for evaluation components
"""
import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.evaluation.model_evaluator import ProductionModelEvaluator, ProductionConstraints, quick_production_check


class TestEvaluation(unittest.TestCase):
    """Test cases for evaluation components"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 1000
        
        # Create realistic test data
        self.y_true = np.random.choice([0, 1], size=self.n_samples, p=[0.95, 0.05])
        
        # Good model predictions (meets constraints)
        self.good_scores = np.random.beta(2, 5, self.n_samples)
        self.good_scores[self.y_true == 1] = np.random.beta(5, 2, np.sum(self.y_true == 1))
        self.good_pred = (self.good_scores > np.percentile(self.good_scores, 90)).astype(int)
        
        # Poor model predictions (doesn't meet constraints)
        self.poor_scores = np.random.uniform(0, 1, self.n_samples)
        self.poor_pred = (self.poor_scores > 0.5).astype(int)
    
    def test_production_constraints(self):
        """Test production constraints configuration"""
        # Test default constraints
        default_constraints = ProductionConstraints()
        self.assertEqual(default_constraints.min_recall, 0.85)
        self.assertEqual(default_constraints.min_f1_score, 0.75)
        self.assertEqual(default_constraints.target_precision, 0.90)
        
        # Test custom constraints
        custom_constraints = ProductionConstraints(
            min_recall=0.90,
            min_f1_score=0.80,
            target_precision=0.95
        )
        self.assertEqual(custom_constraints.min_recall, 0.90)
        self.assertEqual(custom_constraints.min_f1_score, 0.80)
        self.assertEqual(custom_constraints.target_precision, 0.95)
    
    def test_model_evaluation_good_model(self):
        """Test evaluation of a good model that meets constraints"""
        evaluator = ProductionModelEvaluator()
        results = evaluator.evaluate_model(
            self.y_true, self.good_pred, self.good_scores, "Good Model"
        )
        
        # Check basic structure
        self.assertIn('model_name', results)
        self.assertIn('basic_metrics', results)
        self.assertIn('constraint_validation', results)
        self.assertIn('production_ready', results)
        
        # Check that good model is production ready
        self.assertTrue(results['production_ready'])
        
        # Check metrics are reasonable
        metrics = results['basic_metrics']
        self.assertGreaterEqual(metrics['precision'], 0.7)  # Should be decent
        self.assertGreaterEqual(metrics['recall'], 0.7)    # Should be decent
        self.assertGreaterEqual(metrics['f1_score'], 0.7)  # Should be decent
    
    def test_model_evaluation_poor_model(self):
        """Test evaluation of a poor model that doesn't meet constraints"""
        evaluator = ProductionModelEvaluator()
        results = evaluator.evaluate_model(
            self.y_true, self.poor_pred, self.poor_scores, "Poor Model"
        )
        
        # Check that poor model is not production ready
        self.assertFalse(results['production_ready'])
        
        # Check constraint validation details
        constraint_validation = results['constraint_validation']
        self.assertIn('meets_min_recall', constraint_validation)
        self.assertIn('meets_min_f1_score', constraint_validation)
        self.assertIn('meets_target_precision', constraint_validation)
        self.assertIn('overall_pass', constraint_validation)
        self.assertIn('required_improvements', constraint_validation)
        
        # Should have specific improvement recommendations
        self.assertGreater(len(constraint_validation['required_improvements']), 0)
    
    def test_quick_production_check(self):
        """Test quick production check function"""
        # Test good model
        good_result = quick_production_check(self.y_true, self.good_pred, self.good_scores)
        self.assertIn('precision', good_result)
        self.assertIn('recall', good_result)
        self.assertIn('f1_score', good_result)
        self.assertIn('production_ready', good_result)
        self.assertIn('constraint_violations', good_result)
        
        # Test poor model
        poor_result = quick_production_check(self.y_true, self.poor_pred, self.poor_scores)
        self.assertIn('precision', poor_result)
        self.assertIn('recall', poor_result)
        self.assertIn('f1_score', poor_result)
        self.assertIn('production_ready', poor_result)
        self.assertIn('constraint_violations', poor_result)
    
    def test_model_comparison(self):
        """Test comparison of multiple models"""
        evaluator = ProductionModelEvaluator()
        
        # Evaluate multiple models
        good_results = evaluator.evaluate_model(
            self.y_true, self.good_pred, self.good_scores, "Good Model"
        )
        poor_results = evaluator.evaluate_model(
            self.y_true, self.poor_pred, self.poor_scores, "Poor Model"
        )
        
        # Compare models
        comparison = evaluator.compare_models([good_results, poor_results])
        
        # Check comparison structure
        self.assertIn('summary', comparison)
        self.assertIn('detailed_metrics', comparison)
        self.assertIn('recommendations', comparison)
        
        # Check summary
        summary = comparison['summary']
        self.assertIn('total_models', summary)
        self.assertIn('production_ready_models', summary)
        self.assertIn('best_precision_model', summary)
        self.assertIn('best_recall_model', summary)
        self.assertIn('best_f1_model', summary)
        
        # Should have one production ready model
        self.assertEqual(summary['production_ready_models'], 1)
    
    def test_edge_cases(self):
        """Test edge cases in evaluation"""
        evaluator = ProductionModelEvaluator()
        
        # Test with all normal data
        all_normal_true = np.zeros(100)
        all_normal_pred = np.zeros(100)
        all_normal_scores = np.random.uniform(0, 0.1, 100)
        
        results = evaluator.evaluate_model(
            all_normal_true, all_normal_pred, all_normal_scores, "All Normal"
        )
        
        # Should handle without errors
        self.assertIn('basic_metrics', results)
        self.assertIn('constraint_validation', results)
        
        # Test with all anomaly data
        all_anomaly_true = np.ones(100)
        all_anomaly_pred = np.ones(100)
        all_anomaly_scores = np.random.uniform(0.9, 1.0, 100)
        
        results = evaluator.evaluate_model(
            all_anomaly_true, all_anomaly_pred, all_anomaly_scores, "All Anomaly"
        )
        
        # Should handle without errors
        self.assertIn('basic_metrics', results)
        self.assertIn('constraint_validation', results)


if __name__ == '__main__':
    unittest.main()