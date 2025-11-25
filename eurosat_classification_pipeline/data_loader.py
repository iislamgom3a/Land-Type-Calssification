# File: data_loader.py
# Description: Data loading and preprocessing utilities
# ============================================================================

import numpy as np
import os
from skimage.transform import resize
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import subprocess
from config import Config


class DataLoader:
    """Handles loading, preprocessing, and automatic downloading of satellite imagery data"""
    
    def __init__(self, config, kaggle_username=None, kaggle_key=None,DataSize=0):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.DataSize=DataSize

    def download_dataset_if_needed(self, dataset_dir=None):
        """
        Download EuroSAT dataset from Kaggle if not already present
        
        Args:
            dataset_dir: Path to dataset directory (uses config default if None)
        """
        if dataset_dir is None:
            dataset_dir = self.config.DATASET_DIR
            
        parent_dir = os.path.dirname(dataset_dir)
        
        if not os.path.exists(dataset_dir):
            print(f"Dataset not found at {dataset_dir}, downloading from Kaggle...")
            os.makedirs(parent_dir, exist_ok=True)

            # Set Kaggle credentials if provided
            if self.kaggle_username and self.kaggle_key:
                os.environ['KAGGLE_USERNAME'] = self.kaggle_username
                os.environ['KAGGLE_KEY'] = self.kaggle_key
            else:
                print("⚠️ Warning: Kaggle credentials not provided.")
                print("Please ensure kaggle.json is in ~/.kaggle/ or provide credentials.")

            try:
                subprocess.run([
                    "kaggle", "datasets", "download",
                    "-d", self.config.KAGGLE_DATASET,
                    "--unzip",
                    "-p", parent_dir
                ], check=True)
                print("✅ Dataset downloaded successfully.")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to download dataset: {e}")
                print("\n💡 Manual download instructions:")
                print(f"   1. Go to: https://www.kaggle.com/datasets/{self.config.KAGGLE_DATASET}")
                print(f"   2. Download and extract to: {dataset_dir}")
                raise
        else:
            print(f"✅ Dataset found at {dataset_dir}. Skipping download.")

    def load_satellite_data(self, dataset_dir=None,DataSize=0):
        """
        Load satellite images from directory structure
        
        Args:
            dataset_dir: Path to dataset directory (uses config default if None)
            
        Returns:
            tuple: (X, Y) - Images array and labels array
        """
        if dataset_dir is None:
            dataset_dir = self.config.DATASET_DIR
            
        self.download_dataset_if_needed(dataset_dir)
        print("Loading Data...")
        data = []
        labels = []
        
        class_folders = sorted(os.listdir(dataset_dir))
        
        for class_name in class_folders:
            class_path = os.path.join(dataset_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            print(f"Loading class: {class_name}...")
            file_names = os.listdir(class_path)[:DataSize]  # Take only first 1000 images
            for file_name in file_names:
                img_path = os.path.join(class_path, file_name)
                try:
                    img = tiff.imread(img_path)
                    
                    # Validate band count
                    if img.shape[-1] != self.config.NUM_BANDS:
                        print(f"⚠️ Skipping {img_path}: unexpected band count {img.shape[-1]}")
                        continue
                    
                    # Resize and normalize
                    img = resize(
                        img, 
                        (self.config.IMG_SIZE[0], self.config.IMG_SIZE[1], self.config.NUM_BANDS),
                        preserve_range=True,
                        anti_aliasing=True
                    )
                    
                    img = img.astype(np.float32)
                    img /= img.max() if img.max() != 0 else 1.0
                    
                    data.append(img)
                    labels.append(class_name)
                    
                except Exception as e:
                    print(f"⚠️ Skipping {img_path}: {e}")
        
        X = np.array(data)
        Y = np.array(labels)
        
        print(f"\n✅ {len(X)} images loaded successfully.")
        print(f"Data shape: {X.shape}")
        print(f"Classes found: {sorted(set(labels))}")
        
        return X, Y
    
    def prepare_labels(self, Y):
        """
        Encode labels and convert to one-hot format
        
        Args:
            Y: Array of string labels
            
        Returns:
            np.ndarray: One-hot encoded labels
        """
        Y_encoded = self.label_encoder.fit_transform(Y)
        Y_onehot = to_categorical(Y_encoded, num_classes=self.config.NUM_CLASSES)
        print(f"✅ Labels encoded: {len(np.unique(Y_encoded))} unique classes")
        return Y_onehot
    
    def split_data(self, X, Y_onehot, test_size=0.2, val_size=0.5):
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Feature array
            Y_onehot: One-hot encoded labels
            test_size: Proportion for test+val split
            val_size: Proportion of temp set for validation
            
        Returns:
            tuple: (X_train, X_val, X_test, Y_train, Y_val, Y_test)
        """
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X, Y_onehot, 
            test_size=test_size, 
            random_state=self.config.RANDOM_STATE, 
            stratify=Y_onehot
        )
        
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, 
            test_size=val_size, 
            random_state=self.config.RANDOM_STATE, 
            stratify=Y_temp
        )
        
        print(f"✅ Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, Y_train, Y_val, Y_test