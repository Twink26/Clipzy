"""
Dataset Processing Script
Processes raw dataset and prepares it for model training
"""
import sys
import logging
from pathlib import Path

from src.dataset_processor import DatasetProcessor
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main processing function"""
    logger.info("=" * 60)
    logger.info("Dataset Processing Pipeline")
    logger.info("=" * 60)
    
    # Check if dataset exists
    dataset_path = config.DATASET_PATH
    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("\nPlease create a dataset first:")
        logger.info("1. Use collect_dataset.py --interactive")
        logger.info("2. Or manually create data/raw/viral_segments.csv")
        logger.info("3. See data/raw/dataset_template.csv for format")
        sys.exit(1)
    
    try:
        # Initialize processor
        logger.info("Initializing dataset processor...")
        processor = DatasetProcessor()
        
        # Process dataset
        logger.info("\nProcessing dataset...")
        processed_df = processor.process_dataset(
            dataset_path=dataset_path,
            save_path=None  # Uses default processed path
        )
        
        # Split dataset
        logger.info("\nSplitting dataset...")
        train_df, test_df = processor.split_dataset(processed_df)
        
        # Save splits
        train_path = config.PROCESSED_DATA_DIR / "train_features.csv"
        test_path = config.PROCESSED_DATA_DIR / "test_features.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info("\n" + "=" * 60)
        logger.info("Dataset Processing Complete!")
        logger.info("=" * 60)
        logger.info(f"\nProcessed Dataset:")
        logger.info(f"  Total segments: {len(processed_df)}")
        logger.info(f"  Viral segments: {processed_df['is_viral'].sum()}")
        logger.info(f"  Not viral segments: {len(processed_df) - processed_df['is_viral'].sum()}")
        logger.info(f"\nDataset Splits:")
        logger.info(f"  Training: {len(train_df)} segments")
        logger.info(f"  Testing: {len(test_df)} segments")
        logger.info(f"\nOutput Files:")
        logger.info(f"  Processed: {config.PROCESSED_DATA_DIR / 'processed_features.csv'}")
        logger.info(f"  Train: {train_path}")
        logger.info(f"  Test: {test_path}")
        logger.info("\n" + "=" * 60)
        logger.info("Next step: Train the model using nlp_analyzer.py")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        logger.exception("Full error details:")
        sys.exit(1)


if __name__ == '__main__':
    main()

