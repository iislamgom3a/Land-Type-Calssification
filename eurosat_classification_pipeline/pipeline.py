# ============================================================================
# File: pipeline.py
# Description: Main pipeline orchestrator
# ============================================================================

from config import Config
from data_loader import DataLoader
from model import SpectrumNetModel
from evaluator import ModelEvaluator


class EuroSATClassifier:
    """Main pipeline class that orchestrates the entire classification workflow"""
    
    def __init__(self, dataset_path=None, kaggle_username=None, kaggle_key=None,DataSize=0):
        """
        Initialize the classifier
        
        Args:
            dataset_path: Path to dataset (uses config default if None)
            kaggle_username: Kaggle username for dataset download
            kaggle_key: Kaggle API key for dataset download
        """
        self.config = Config()
        self.dataset_path = dataset_path if dataset_path else self.config.DATASET_DIR
        self.data_loader = DataLoader(self.config, kaggle_username, kaggle_key,DataSize)
        self.model_builder = None
        self.evaluator = None
        self.DataSize=DataSize
        
        # Data storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.Y_train = None
        self.Y_val = None
        self.Y_test = None
        
    def load_and_prepare_data(self):
        """Load data and prepare for training"""
        print(f"\n{'='*60}")
        print("📂 LOADING AND PREPARING DATA")
        print(f"{'='*60}\n")
        
        # Load raw data
        X, Y = self.data_loader.load_satellite_data(self.dataset_path,self.DataSize)
        
        # Encode labels
        Y_onehot = self.data_loader.prepare_labels(Y)
        
        # Split data
        self.X_train, self.X_val, self.X_test, \
        self.Y_train, self.Y_val, self.Y_test = self.data_loader.split_data(X, Y_onehot)
        
        print(f"\n✅ Data loading and preparation completed!")
        
    def build_and_train(self, epochs=None, batch_size=None):
        """
        Build model and train
        
        Args:
            epochs: Number of epochs (uses config default if None)
            batch_size: Batch size (uses config default if None)
        """
        print(f"\n{'='*60}")
        print("🏗️ BUILDING AND TRAINING MODEL")
        print(f"{'='*60}\n")
        
        # Build model
        self.model_builder = SpectrumNetModel(self.config)
        self.model_builder.build_model(num_classes=self.config.NUM_CLASSES)
        self.model_builder.model.summary()
        
        # Compile
        self.model_builder.compile_model()
        
        # Train
        self.model_builder.train(
            self.X_train, self.Y_train,
            self.X_val, self.Y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
    def evaluate_model(self, detailed_report=True):
        """
        Evaluate trained model
        
        Args:
            detailed_report: Whether to print detailed classification report
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\n{'='*60}")
        print("🎯 EVALUATING MODEL")
        print(f"{'='*60}\n")
        
        # Load best model
        best_model = self.model_builder.load_best_model()
        
        # Create evaluator
        self.evaluator = ModelEvaluator(best_model)
        
        # Evaluate
        self.evaluator.evaluate(self.X_test, self.Y_test)
        
        # Predict
        predictions = self.evaluator.predict(self.X_test)
        
        # Calculate metrics
        metrics, y_true, y_pred = self.evaluator.calculate_metrics(self.Y_test, predictions)
        
        # Print results
        total_params = self.model_builder.get_model_params()
        self.evaluator.print_metrics(
            metrics, 
            total_params, 
            self.model_builder.training_time,
            class_names=self.config.CLASSES
        )
        
        # Print detailed report if requested
        if detailed_report:
            self.evaluator.print_classification_report(
                y_true, y_pred, 
                class_names=self.config.CLASSES
            )
        
        return metrics
    
    def run_full_pipeline(self, epochs=None, batch_size=None, detailed_report=True):
        """
        Execute complete pipeline from data loading to evaluation
        
        Args:
            epochs: Number of epochs (uses config default if None)
            batch_size: Batch size (uses config default if None)
            detailed_report: Whether to print detailed classification report
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "="*60)
        print("🚀 STARTING EUROSAT CLASSIFICATION PIPELINE")
        print("="*60)
        
        try:
            self.load_and_prepare_data()
            self.build_and_train(epochs=epochs, batch_size=batch_size)
            metrics = self.evaluate_model(detailed_report=detailed_report)
            
            print("\n" + "="*60)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60 + "\n")
            
            return metrics
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            raise
