#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI Generated Talking Face Video Evaluation Dataset Quality Analysis Report

Comprehensive analysis of EmotionTalk dataset quality, feature distributions, and anomalies
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset_quality(dataset_path):
    """
    Analyze dataset quality and generate comprehensive report
    
    Args:
        dataset_path: Path to the dataset file
    """
    print("=" * 80)
    print("AI Generated Talking Face Video Evaluation Dataset Quality Analysis")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Handle different data formats
    if isinstance(data, dict):
        train_data = data.get('train', [])
        val_data = data.get('val', [])
        test_data = data.get('test', [])
    elif isinstance(data, list):
        train_data = [item for item in data if item.get('split') == 'train']
        val_data = [item for item in data if item.get('split') == 'valid']
        test_data = [item for item in data if item.get('split') == 'test']
    else:
        print(f"Unsupported data format: {type(data)}")
        return None
    
    total_samples = len(train_data) + len(val_data) + len(test_data)
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {total_samples}")
    print(f"Training set: {len(train_data)} samples ({len(train_data)/total_samples*100:.1f}%)")
    print(f"Validation set: {len(val_data)} samples ({len(val_data)/total_samples*100:.1f}%)")
    print(f"Test set: {len(test_data)} samples ({len(test_data)/total_samples*100:.1f}%)")
    print()
    
    # Analysis results storage
    analysis_results = {}
    
    # 1. Label Distribution Analysis
    print("=" * 80)
    print("1. LABEL DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    all_labels = []
    for split_data, split_name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
        if split_data:
            for sample in split_data:
                if 'labels' in sample:
                    labels = sample['labels'].copy()
                    labels['split'] = split_name
                    all_labels.append(labels)
    
    if all_labels:
        labels_df = pd.DataFrame(all_labels)
        score_columns = ['lip_sync_score', 'expression_score', 'audio_quality_score', 
                        'cross_modal_score', 'overall_score']
        
        print(f"Total labeled samples: {len(labels_df)}")
        print()
        
        for col in score_columns:
            if col in labels_df.columns:
                valid_mask = labels_df[col] != -1.0
                valid_count = valid_mask.sum()
                invalid_count = (~valid_mask).sum()
                
                print(f"{col}:")
                print(f"  Valid samples: {valid_count} ({valid_count/len(labels_df)*100:.1f}%)")
                print(f"  Invalid samples (-1.0): {invalid_count} ({invalid_count/len(labels_df)*100:.1f}%)")
                
                if valid_count > 0:
                    valid_values = labels_df[col][valid_mask]
                    print(f"  Mean: {valid_values.mean():.3f}")
                    print(f"  Std: {valid_values.std():.3f}")
                    print(f"  Min: {valid_values.min():.3f}")
                    print(f"  Max: {valid_values.max():.3f}")
                    print(f"  Median: {valid_values.median():.3f}")
                print()
        
        analysis_results['labels_df'] = labels_df
    
    # 2. Feature Quality Analysis
    print("=" * 80)
    print("2. FEATURE QUALITY ANALYSIS")
    print("=" * 80)
    
    feature_dims = {
        'visual': 163,
        'audio': 768,
        'keypoint': 1404,
        'au': 17
    }
    
    feature_stats = {}
    
    for feature_type in ['visual', 'audio', 'keypoint', 'au']:
        print(f"{feature_type.upper()} Features:")
        print("-" * 40)
        
        all_features = []
        nan_count = 0
        inf_count = 0
        
        # Collect feature data (limit to 100 samples for efficiency)
        for sample in train_data[:100]:
            if 'features' in sample and feature_type in sample['features']:
                features = np.array(sample['features'][feature_type])
                
                # Check for NaN and Inf
                nan_count += np.isnan(features).sum()
                inf_count += np.isinf(features).sum()
                
                all_features.append(features)
        
        if not all_features:
            print(f"  No {feature_type} feature data found")
            continue
        
        try:
            features_array = np.stack(all_features)
            print(f"  Feature shape: {features_array.shape}")
            print(f"  NaN values: {nan_count}")
            print(f"  Inf values: {inf_count}")
            
            # Calculate basic statistics
            flat_features = features_array.reshape(-1, features_array.shape[-1])
            
            # Check for zero variance features
            feature_vars = np.var(flat_features, axis=0)
            zero_var_features = np.sum(feature_vars == 0)
            print(f"  Zero variance features: {zero_var_features}/{len(feature_vars)}")
            
            # Calculate statistics
            feature_stats[feature_type] = {
                'mean': np.mean(flat_features, axis=0),
                'std': np.std(flat_features, axis=0),
                'min': np.min(flat_features, axis=0),
                'max': np.max(flat_features, axis=0),
                'median': np.median(flat_features, axis=0)
            }
            
            print(f"  Feature value range: [{np.min(flat_features):.3f}, {np.max(flat_features):.3f}]")
            
            # Quality flags
            if nan_count > 0:
                print(f"  WARNING: Found {nan_count} NaN values!")
            if zero_var_features > 0:
                print(f"  INFO: Found {zero_var_features} zero-variance features")
            
        except Exception as e:
            print(f"  Error in feature analysis: {e}")
        
        print()
    
    analysis_results['feature_stats'] = feature_stats
    
    # 3. Feature Scaling Analysis
    print("=" * 80)
    print("3. FEATURE SCALING ANALYSIS")
    print("=" * 80)
    
    scaling_issues = {}
    
    for feature_type in ['visual', 'audio', 'keypoint', 'au']:
        print(f"{feature_type.upper()} Feature Scaling:")
        print("-" * 40)
        
        sample_features = []
        for sample in train_data[:50]:  # Limit samples
            if 'features' in sample and feature_type in sample['features']:
                features = np.array(sample['features'][feature_type])
                # Average over time dimension for analysis
                if len(features.shape) == 2:
                    features = np.mean(features, axis=0)
                sample_features.append(features)
        
        if not sample_features:
            continue
        
        try:
            features_matrix = np.stack(sample_features)
            
            # Calculate statistics per feature dimension
            feature_means = np.mean(features_matrix, axis=0)
            feature_stds = np.std(features_matrix, axis=0)
            feature_ranges = np.max(features_matrix, axis=0) - np.min(features_matrix, axis=0)
            
            print(f"  Feature dimensions: {features_matrix.shape[1]}")
            print(f"  Mean range: [{np.min(feature_means):.3f}, {np.max(feature_means):.3f}]")
            print(f"  Std range: [{np.min(feature_stds):.3f}, {np.max(feature_stds):.3f}]")
            print(f"  Range: [{np.min(feature_ranges):.3f}, {np.max(feature_ranges):.3f}]")
            
            # Check for scaling issues
            mean_range_ratio = np.max(np.abs(feature_means)) / (np.min(np.abs(feature_means)) + 1e-8)
            std_range_ratio = np.max(feature_stds) / (np.min(feature_stds) + 1e-8)
            range_ratio = np.max(feature_ranges) / (np.min(feature_ranges) + 1e-8)
            
            print(f"  Mean range ratio: {mean_range_ratio:.1f}")
            print(f"  Std range ratio: {std_range_ratio:.1f}")
            print(f"  Range ratio: {range_ratio:.1f}")
            
            # Determine if scaling is needed
            needs_scaling = (mean_range_ratio > 100 or std_range_ratio > 100 or range_ratio > 1000)
            scaling_issues[feature_type] = needs_scaling
            
            if needs_scaling:
                print(f"  STATUS: NEEDS SCALING")
            else:
                print(f"  STATUS: OK")
                
        except Exception as e:
            print(f"  Error in scaling analysis: {e}")
        
        print()
    
    analysis_results['scaling_issues'] = scaling_issues
    
    # 4. Data Quality Issues Summary
    print("=" * 80)
    print("4. DATA QUALITY ISSUES SUMMARY")
    print("=" * 80)
    
    issues_found = []
    
    # Check for NaN values
    for feature_type in ['visual', 'audio', 'keypoint', 'au']:
        if feature_type in feature_stats:
            nan_count = 0
            for sample in train_data[:50]:
                if 'features' in sample and feature_type in sample['features']:
                    features = np.array(sample['features'][feature_type])
                    nan_count += np.isnan(features).sum()
            if nan_count > 0:
                issues_found.append(f"{feature_type}: {nan_count} NaN values")
    
    # Check for invalid labels
    if 'labels_df' in analysis_results:
        labels_df = analysis_results['labels_df']
        for col in ['expression_score', 'audio_quality_score', 'cross_modal_score', 'overall_score']:
            if col in labels_df.columns:
                invalid_ratio = (labels_df[col] == -1.0).mean()
                if invalid_ratio > 0.1:
                    issues_found.append(f"{col}: {invalid_ratio*100:.1f}% invalid labels")
    
    # Check for scaling issues
    for feature_type, needs_scaling in scaling_issues.items():
        if needs_scaling:
            issues_found.append(f"{feature_type}: severe scaling issues detected")
    
    if issues_found:
        print("ISSUES DETECTED:")
        for i, issue in enumerate(issues_found, 1):
            print(f"{i}. {issue}")
    else:
        print("No major issues detected")
    
    print()
    
    # 5. Recommendations
    print("=" * 80)
    print("5. RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    # Label-related recommendations
    if any('invalid labels' in issue for issue in issues_found):
        recommendations.append("Handle invalid labels (-1.0): Consider removing samples or using imputation")
    
    # Feature scaling recommendations
    scaling_features = [ft for ft, needs in scaling_issues.items() if needs]
    if scaling_features:
        recommendations.append(f"Apply feature scaling to: {', '.join(scaling_features)}")
    
    # NaN handling
    if any('NaN' in issue for issue in issues_found):
        recommendations.append("Handle NaN values: Consider imputation or feature removal")
    
    # Dimensionality recommendations
    recommendations.append("Consider dimensionality reduction for high-dimensional features (keypoint: 1404D)")
    
    # Model performance recommendations
    recommendations.append("For poor Cross Modal performance: Consider attention mechanisms or specialized fusion strategies")
    recommendations.append("For low R² (0.174): Consider data augmentation, regularization, or model architecture changes")
    
    print("RECOMMENDED ACTIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print()
    
    # 6. Feature Dimensionality Analysis
    print("=" * 80)
    print("6. FEATURE DIMENSIONALITY ANALYSIS")
    print("=" * 80)
    
    print("Feature Type | Dimensions | Proportion | Status")
    print("-" * 50)
    
    total_dims = sum(feature_dims.values())
    for feature_type, dims in feature_dims.items():
        proportion = dims / total_dims * 100
        status = "High" if dims > 1000 else "Medium" if dims > 100 else "Low"
        print(f"{feature_type:11} | {dims:10} | {proportion:9.1f}% | {status}")
    
    print(f"{'Total':11} | {total_dims:10} | {'100.0%':>9} |")
    print()
    
    # Identify dimension imbalance
    max_dims = max(feature_dims.values())
    min_dims = min(feature_dims.values())
    imbalance_ratio = max_dims / min_dims
    
    print(f"Dimension imbalance ratio: {imbalance_ratio:.1f} (max: {max_dims}, min: {min_dims})")
    if imbalance_ratio > 50:
        print("WARNING: Severe dimension imbalance detected!")
        print("Consider: PCA for keypoint features, feature selection, or weighted fusion")
    
    print()
    
    return analysis_results

def save_report(results, output_path):
    """Save analysis results to file"""
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Analysis report saved to: {output_path}")

def main():
    """Main function"""
    # Dataset paths
    possible_paths = [
        "f:/bs/datasets/ch_sims_final_dataset.pkl",
        "f:/bs/datasets/ch_sims_processed_data_cache_1985.pkl",
        "f:/bs/datasets/ac.pkl"
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if not dataset_path:
        print("No dataset file found!")
        return
    
    print(f"Using dataset: {dataset_path}")
    print()
    
    # Run analysis
    results = analyze_dataset_quality(dataset_path)
    
    if results:
        # Save results
        output_path = "f:/bs/dataset_quality_report.pkl"
        save_report(results, output_path)
        
        print("=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

if __name__ == "__main__":
    main()