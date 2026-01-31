#!/usr/bin/env python3
"""
Master Training Script
Trains all three models: Isolation Forest, LSTM Autoencoder, and GRU Autoencoder
"""

import os
import logging
import subprocess
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_training_script(script_name):
    """Run a training script and return results"""
    logger.info(f"Running {script_name}...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", script_name],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {script_name} completed successfully")
            return {
                'status': 'success',
                'output': result.stdout,
                'error': None
            }
        else:
            logger.error(f"❌ {script_name} failed")
            logger.error(f"Error: {result.stderr}")
            return {
                'status': 'failed',
                'output': result.stdout,
                'error': result.stderr
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to run {script_name}: {e}")
        return {
            'status': 'failed',
            'output': '',
            'error': str(e)
        }

def train_all_models():
    """Train all three models"""
    logger.info("Starting training of all models...")
    logger.info("=" * 50)
    
    # Training scripts to run
    training_scripts = [
        "scripts.train_isolation_forest",
        "scripts.train_lstm_autoencoder", 
        "scripts.train_gru_autoencoder"
    ]
    
    results = {}
    
    # Run each training script
    for script in training_scripts:
        script_name = script.split('.')[-1].replace('train_', '').replace('_', ' ').title()
        logger.info(f"\n🚀 Training {script_name}...")
        
        result = run_training_script(script)
        results[script_name] = result
        
        if result['status'] == 'success':
            logger.info(f"✅ {script_name} training completed")
        else:
            logger.error(f"❌ {script_name} training failed")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 50)
    
    successful_trainings = sum(1 for result in results.values() if result['status'] == 'success')
    total_trainings = len(results)
    
    logger.info(f"Successful trainings: {successful_trainings}/{total_trainings}")
    
    for model_name, result in results.items():
        status = "SUCCESS" if result['status'] == 'success' else "FAILED"
        logger.info(f"{model_name}: {status}")
        
        if result['status'] == 'success':
            # Extract performance metrics from output
            lines = result['output'].split('\n')
            for line in lines:
                if 'Performance' in line and '-' in line:
                    logger.info(f"  {line.strip()}")
    
    # Overall status
    if successful_trainings == total_trainings:
        logger.info("\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        logger.info("Models are ready for inference and deployment.")
        return True
    else:
        logger.error(f"\n⚠️  {total_trainings - successful_trainings} models failed to train.")
        logger.error("Check the error messages above for details.")
        return False

if __name__ == "__main__":
    start_time = datetime.now()
    success = train_all_models()
    end_time = datetime.now()
    
    duration = end_time - start_time
    logger.info(f"\nTotal training time: {duration}")
    
    if success:
        print("\n✅ Training pipeline completed successfully!")
        print("All models are now available in the models_saved/ directory.")
        sys.exit(0)
    else:
        print("\n❌ Training pipeline had failures.")
        sys.exit(1)