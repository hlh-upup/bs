#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Preprocessing Pipeline for AI Generated Talking Face Video Evaluation

Implements the recommendations from data quality analysis:
1. Handle NaN values and invalid labels
2. Feature scaling and normalization
3. Dimensionality reduction for keypoint features
4. Data augmentation techniques
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, config=None):
        """
        Initialize data preprocessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scalers = {}
        self.pca_models = {}
        self.imputers = {}
        self.label_scalers = {}
        
        # Default configuration
        self.default_config = {
            'handle_nan': 'impute',  # 'remove', 'impute', 'zero'
            'scaling_method': 'standard',  # 'standard', 'minmax', 'robust'
            'pca_components': {
                'keypoint': 200,  # Reduce from 1404 to 200
                'visual': None,   # Keep original
                'audio': None,    # Keep original
                'au': None        # Keep original
            },
            'label_imputation': 'mean',  # 'mean', 'median', 'interpolation'
            'augmentation': {
                'enabled': True,
                'noise_level': 0.01,
                'temporal_warp': True
            }
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
    def clean_features(self, features_dict):
        """
        Clean features by handling NaN values and outliers
        
        Args:
            features_dict: Dictionary of feature arrays
            
        Returns:
            Cleaned features dictionary
        """
        cleaned_features = {}
        
        for feature_type, features in features_dict.items():
            if feature_type not in ['visual', 'audio', 'keypoint', 'au']:
                cleaned_features[feature_type] = features
                continue
                
            # Convert to numpy array
            features_array = np.array(features, dtype=np.float32)
            
            # Handle NaN values
            if np.isnan(features_array).any():
                if self.config['handle_nan'] == 'impute':
                    # Use median imputation per feature dimension
                    original_shape = features_array.shape
                    flat_features = features_array.reshape(-1, features_array.shape[-1])
                    
                    if feature_type not in self.imputers:
                        self.imputers[feature_type] = SimpleImputer(strategy='median')
                        # Fit imputer on available data
                        self.imputers[feature_type].fit(flat_features)
                    
                    # Apply imputation
                    features_array = self.imputers[feature_type].transform(flat_features).reshape(original_shape)
                    
                elif self.config['handle_nan'] == 'zero':
                    features_array = np.nan_to_num(features_array, nan=0.0)
                elif self.config['handle_nan'] == 'remove':
                    # Remove time steps with NaN
                    if len(features_array.shape) == 3:
                        valid_mask = ~np.isnan(features_array).any(axis=1)
                        features_array = features_array[valid_mask]
            
            cleaned_features[feature_type] = features_array
            
        return cleaned_features
    
    def scale_features(self, features_dict, fit_scalers=True):
        """
        Apply feature scaling
        
        Args:
            features_dict: Dictionary of feature arrays
            fit_scalers: Whether to fit scalers (True for training, False for inference)
            
        Returns:
            Scaled features dictionary
        """
        scaled_features = {}
        
        for feature_type, features in features_dict.items():
            if feature_type not in ['visual', 'audio', 'keypoint', 'au']:
                scaled_features[feature_type] = features
                continue
            
            features_array = np.array(features, dtype=np.float32)
            original_shape = features_array.shape
            
            # Reshape for scaling (flatten time dimension)
            if len(original_shape) == 3:  # (time, features)
                features_2d = features_array.reshape(-1, original_shape[-1])
            else:
                features_2d = features_array.reshape(-1, 1) if len(features_array.shape) == 1 else features_array
            
            # Initialize scaler if needed
            if feature_type not in self.scalers:
                if self.config['scaling_method'] == 'standard':
                    self.scalers[feature_type] = StandardScaler()
                elif self.config['scaling_method'] == 'minmax':
                    self.scalers[feature_type] = MinMaxScaler()
                elif self.config['scaling_method'] == 'robust':
                    self.scalers[feature_type] = RobustScaler()
            
            # Fit and transform
            if fit_scalers:
                features_scaled = self.scalers[feature_type].fit_transform(features_2d)
            else:
                features_scaled = self.scalers[feature_type].transform(features_2d)
            
            # Reshape back to original shape
            if len(original_shape) == 3:
                features_scaled = features_scaled.reshape(original_shape)
            
            scaled_features[feature_type] = features_scaled
        
        return scaled_features
    
    def apply_dimensionality_reduction(self, features_dict, fit_models=True):
        """
        Apply PCA for dimensionality reduction
        
        Args:
            features_dict: Dictionary of feature arrays
            fit_models: Whether to fit PCA models (True for training, False for inference)
            
        Returns:
            Features with reduced dimensionality
        """
        reduced_features = {}
        
        for feature_type, features in features_dict.items():
            n_components = self.config['pca_components'].get(feature_type)
            
            if n_components is None or feature_type not in ['keypoint', 'visual']:
                reduced_features[feature_type] = features
                continue
            
            features_array = np.array(features, dtype=np.float32)
            original_shape = features_array.shape
            
            # Reshape for PCA
            if len(original_shape) == 3:
                features_2d = features_array.reshape(-1, original_shape[-1])
            else:
                features_2d = features_array
            
            # Initialize PCA model if needed
            if feature_type not in self.pca_models:
                self.pca_models[feature_type] = PCA(n_components=n_components, random_state=42)
            
            # Apply PCA
            if fit_models:
                features_reduced = self.pca_models[feature_type].fit_transform(features_2d)
                print(f"Applied PCA to {feature_type}: {original_shape[-1]} -> {n_components} components")
                print(f"Explained variance ratio: {self.pca_models[feature_type].explained_variance_ratio_.sum():.3f}")
            else:
                features_reduced = self.pca_models[feature_type].transform(features_2d)
            
            # Reshape back
            if len(original_shape) == 3:
                new_shape = (original_shape[0], original_shape[1], n_components)
                features_reduced = features_reduced.reshape(new_shape)
            
            reduced_features[feature_type] = features_reduced
        
        return reduced_features
    
    def handle_invalid_labels(self, labels_dict):
        """
        Handle invalid labels (-1.0 values)
        
        Args:
            labels_dict: Dictionary of label values
            
        Returns:
            Cleaned labels dictionary
        """
        cleaned_labels = labels_dict.copy()
        
        for label_name, label_value in labels_dict.items():
            if label_value == -1.0:
                if self.config['label_imputation'] == 'mean':
                    # Use mean of valid labels (should be calculated from training data)
                    cleaned_labels[label_name] = 0.0  # Placeholder, should be calculated
                elif self.config['label_imputation'] == 'median':
                    cleaned_labels[label_name] = 0.0  # Placeholder
                else:
                    cleaned_labels[label_name] = 0.0  # Default to 0
        
        return cleaned_labels
    
    def apply_temporal_augmentation(self, features_dict):
        """
        Apply temporal data augmentation
        
        Args:
            features_dict: Dictionary of feature arrays
            
        Returns:
            Augmented features dictionary
        """
        if not self.config['augmentation']['enabled']:
            return features_dict
        
        augmented_features = {}
        
        for feature_type, features in features_dict.items():
            if feature_type not in ['visual', 'audio', 'keypoint', 'au']:
                augmented_features[feature_type] = features
                continue
            
            features_array = np.array(features, dtype=np.float32)
            
            # Add Gaussian noise
            if self.config['augmentation'].get('noise_level', 0) > 0:
                noise = np.random.normal(0, self.config['augmentation']['noise_level'], 
                                       features_array.shape)
                features_array = features_array + noise
            
            # Temporal warping (simple version)
            if self.config['augmentation'].get('temporal_warp', False) and len(features_array.shape) == 3:
                # Random time stretching/shrinking
                time_steps = features_array.shape[0]
                if time_steps > 1:
                    warp_factor = np.random.uniform(0.9, 1.1)
                    new_time_steps = int(time_steps * warp_factor)
                    
                    # Simple linear interpolation
                    if new_time_steps != time_steps:
                        from scipy import interpolate
                        old_time = np.arange(time_steps)
                        new_time = np.linspace(0, time_steps - 1, new_time_steps)
                        
                        warped_features = np.zeros((new_time_steps, features_array.shape[1]), dtype=np.float32)
                        for i in range(features_array.shape[1]):
                            f = interpolate.interp1d(old_time, features_array[:, i], 
                                                   kind='linear', fill_value='extrapolate')
                            warped_features[:, i] = f(new_time)
                        
                        features_array = warped_features
            
            augmented_features[feature_type] = features_array
        
        return augmented_features
    
    def preprocess_sample(self, sample, fit_models=True):
        """
        Preprocess a single sample
        
        Args:
            sample: Dictionary containing features and labels
            fit_models: Whether to fit preprocessing models
            
        Returns:
            Preprocessed sample
        """
        processed_sample = sample.copy()
        
        # Clean features
        if 'features' in sample:
            features = self.clean_features(sample['features'])
            features = self.scale_features(features, fit_models)
            features = self.apply_dimensionality_reduction(features, fit_models)
            
            if fit_models:  # Only augment training data
                features = self.apply_temporal_augmentation(features)
            
            processed_sample['features'] = features
        
        # Handle invalid labels
        if 'labels' in sample:
            labels = self.handle_invalid_labels(sample['labels'])
            processed_sample['labels'] = labels
        
        return processed_sample
    
    def preprocess_dataset(self, data, fit_models=True):
        """
        Preprocess entire dataset
        
        Args:
            data: Dataset (list of samples or dict with train/val/test)
            fit_models: Whether to fit preprocessing models
            
        Returns:
            Preprocessed dataset
        """
        print("Starting data preprocessing...")
        
        if isinstance(data, dict):
            # Handle dict format with train/val/test splits
            processed_data = {}
            for split_name, split_data in data.items():
                print(f"Processing {split_name} split...")
                processed_split = []
                for sample in split_data:
                    processed_sample = self.preprocess_sample(sample, fit_models)
                    processed_split.append(processed_sample)
                processed_data[split_name] = processed_split
                print(f"  Processed {len(processed_split)} samples")
            
            return processed_data
        
        elif isinstance(data, list):
            # Handle list format
            processed_data = []
            for sample in data:
                processed_sample = self.preprocess_sample(sample, fit_models)
                processed_data.append(processed_sample)
            
            print(f"Processed {len(processed_data)} samples")
            return processed_data
        
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
    
    def save_preprocessor(self, filepath):
        """Save preprocessor state"""
        state = {
            'config': self.config,
            'scalers': self.scalers,
            'pca_models': self.pca_models,
            'imputers': self.imputers,
            'label_scalers': self.label_scalers
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load preprocessor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.scalers = state['scaleras']
        self.pca_models = state['pca_models']
        self.imputers = state['imputers']
        self.label_scalers = state['label_scalers']
        print(f"Preprocessor loaded from {filepath}")

def main():
    """Main function for testing the preprocessing pipeline"""
    
    # Load dataset
    print("Loading dataset...")
    dataset_path = "f:/bs/datasets/ch_sims_processed_data_cache_1985.pkl"
    
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    # Initialize preprocessor
    config = {
        'handle_nan': 'impute',
        'scaling_method': 'standard',
        'pca_components': {
            'keypoint': 50,  # Reduced from 200 to 50 (less than min_samples)
            'visual': None,
            'audio': None,
            'au': None
        },
        'label_imputation': 'mean',
        'augmentation': {
            'enabled': True,
            'noise_level': 0.01,
            'temporal_warp': False  # Disable for now
        }
    }
    
    preprocessor = DataPreprocessor(config)
    
    # Preprocess training data
    print("\nPreprocessing training data...")
    train_data = [item for item in data if item.get('split') == 'train']
    processed_train = preprocessor.preprocess_dataset(train_data[:100], fit_models=True)  # Test with 100 samples
    
    # Preprocess validation data (using fitted models)
    print("\nPreprocessing validation data...")
    val_data = [item for item in data if item.get('split') == 'valid']
    processed_val = preprocessor.preprocess_dataset(val_data[:50], fit_models=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor("f:/bs/preprocessor_state.pkl")
    
    # Save processed data
    processed_data = {
        'train': processed_train,
        'val': processed_val,
        'config': config
    }
    
    output_path = "f:/bs/processed_dataset.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\nProcessed dataset saved to {output_path}")
    print("Preprocessing pipeline completed!")

if __name__ == "__main__":
    main()