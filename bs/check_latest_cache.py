import os
import pickle
import pandas as pd

def analyze_latest_cache(cache_file_path):
    """
    Analyzes the specified cache file to determine the quantity and quality of the data.

    Args:
        cache_file_path (str): The absolute path to the .pkl cache file.
    """
    print(f"--- Analyzing Cache File: {os.path.basename(cache_file_path)} ---")

    # 1. Check if the file exists
    if not os.path.exists(cache_file_path):
        print(f"Error: Cache file not found at '{cache_file_path}'")
        return

    print(f"File found. Size: {os.path.getsize(cache_file_path) / (1024*1024):.2f} MB")

    # 2. Load the data
    try:
        with open(cache_file_path, 'rb') as f:
            data = pickle.load(f)
        print("Successfully loaded the pickle file.")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    # 3. Analyze the content
    if not isinstance(data, list) or not data:
        print("Error: Expected a non-empty list of samples, but found something else.")
        return

    total_samples = len(data)
    print(f"\n[Analysis Results]")
    print(f"Total number of samples (videos) in this cache: {total_samples}")

    # Inspect the first sample
    first_sample = data[0]
    print("\n--- Structure of the first sample ---")
    if isinstance(first_sample, dict):
        for key, value in first_sample.items():
            if hasattr(value, 'shape'):
                print(f"  - Key: '{key}', Type: {type(value)}, Shape: {value.shape}")
            else:
                print(f"  - Key: '{key}', Type: {type(value)}, Value: {value}")
    else:
        print(f"First sample is of type: {type(first_sample)}")

    # Check for uniqueness
    video_ids = [item.get('video_id', None) for item in data]
    unique_video_ids = set(video_ids)
    num_unique_videos = len(unique_video_ids)
    num_duplicates = total_samples - num_unique_videos

    print("\n--- Data Uniqueness ---")
    print(f"Number of unique video IDs: {num_unique_videos}")
    print(f"Number of duplicate entries: {num_duplicates}")

    if num_duplicates > 0:
        print("Warning: Duplicate entries were found. This is expected due to the previous caching strategy.")
    else:
        print("Good news: All entries in this cache file are unique.")

    # 4. Conclusion and Recommendation
    print("\n--- Conclusion ---")
    if total_samples > 0:
        print(f"This cache file contains {total_samples} samples, with {num_unique_videos} unique videos.")
        print("This data should be sufficient to create a dataset for training.")
        print("Next step recommendation: Use this file to create a final, compressed training dataset.")
    else:
        print("The cache file is empty or invalid. Further investigation is needed.")

if __name__ == '__main__':
    # Path to the latest cache file provided by the user
    LATEST_CACHE_FILE = r'f:\bs\datasets\ch_sims_processed_data_cache_1985.pkl'
    analyze_latest_cache(LATEST_CACHE_FILE)