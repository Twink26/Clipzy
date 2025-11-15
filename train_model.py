"""
Model Training Script
Trains the viral segment detection model on processed dataset
"""
import sys
import logging
from pathlib import Path

from src.nlp_analyzer import NLPAnalyzer
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("Model Training Pipeline")
    logger.info("=" * 60)
    
    # Check if processed dataset exists
    processed_path = config.PROCESSED_DATA_DIR / "processed_features.csv"
    if not processed_path.exists():
        logger.error(f"Processed dataset not found: {processed_path}")
        logger.info("\nPlease process the dataset first:")
        logger.info("  python process_dataset.py")
        sys.exit(1)
    
    try:
        # Initialize analyzer
        logger.info("Initializing NLP Analyzer...")
        analyzer = NLPAnalyzer()
        
        # Train model
        logger.info("\nTraining model...")
        metrics = analyzer.train(
            processed_dataset_path=str(processed_path),
            save_model=True
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("Model Training Complete!")
        logger.info("=" * 60)
        logger.info(f"\nModel Performance:")
        logger.info(f"  Train Accuracy: {metrics['train_accuracy']:.3f}")
        logger.info(f"  Test Accuracy: {metrics['test_accuracy']:.3f}")
        logger.info(f"  Test Precision: {metrics['test_precision']:.3f}")
        logger.info(f"  Test Recall: {metrics['test_recall']:.3f}")
        logger.info(f"  Test F1 Score: {metrics['test_f1']:.3f}")
        logger.info(f"  CV F1 Score: {metrics['cv_f1_mean']:.3f} (+/- {metrics['cv_f1_std']:.3f})")
        logger.info(f"\nModel saved to: {config.MODEL_PATH}")
        logger.info("\n" + "=" * 60)
        logger.info("Model is ready to use!")
        logger.info("You can now generate reels using main.py")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.exception("Full error details:")
        sys.exit(1)


if __name__ == '__main__':
    main()

