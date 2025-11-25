# ============================================================================
# File: main.py
# Description: Entry point for running the classification pipeline
# ============================================================================

from pipeline import EuroSATClassifier


def main():
    """Main function to run EuroSAT classification"""
    
    # Configuration
    DATASET_PATH = "./EuroSAT/EuroSATallBands"  # Local path
    KAGGLE_USERNAME = None  # Set your Kaggle username or None
    KAGGLE_KEY = None  # Set your Kaggle API key or None
    
    # Training parameters (None = use config defaults)
    EPOCHS = None  # or specify like: 40
    BATCH_SIZE = None  # or specify like: 16
    
    print("\n" + "="*60)
    print("🌍 EUROSAT SATELLITE IMAGE CLASSIFICATION")
    print("="*60)
    
    # Create classifier instance
    classifier = EuroSATClassifier(
        dataset_path=DATASET_PATH,
        kaggle_username=KAGGLE_USERNAME,
        kaggle_key=KAGGLE_KEY,
        DataSize=500,
    )
    
    # Run full pipeline
    metrics = classifier.run_full_pipeline(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        detailed_report=True
    )
    
    return metrics


if __name__ == "__main__":
    main()
