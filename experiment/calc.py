import os
import pandas as pd
import re



def analyze_model_performance(directory, datasets, model=''):
    performance_data = {}
    # First, collect all the data
    for dataset in datasets:
        json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json') and dataset in f and model in f])
        for filename in json_files:
            with open(os.path.join(directory, filename), 'r') as f:
                data = pd.read_json(f)
            total = len(data['flag'])
            flagged = sum(data['flag'])
            ratio = flagged / total if total > 0 else 0
            # Extract model name from filename            
            model_name = filename.replace(f'-{dataset}.json', '')
            if model_name not in performance_data:
                performance_data[model_name] = {}
            performance_data[model_name][dataset] = ratio
    # Then, print results for each model
    for model_name, model_data in performance_data.items():
        print(f"\nModel: {model_name}")
        print("-" * 90)
        print(f"{'Dataset':<20} {'Ratio':<10}")
        print("-" * 90)
        total_ratio = 0
        count = 0
        for dataset, ratio in model_data.items():
            print(f"{dataset:<20} {ratio:.4f}")
            total_ratio += ratio
            count += 1
        print("-" * 90)
        for dataset, ratio in model_data.items():
            print(f"& {ratio*100:.2f} ",end="")
        avg_ratio = total_ratio / count if count > 0 else 0
        print(f"& {avg_ratio*100:.2f} \\\\")
        print("-" * 90)
        print(f"{'Average':<20} {avg_ratio:.4f}")
        print("\n")

# List of datasets
datasets = ['boolq', 'piqa', 'social_i_qa', 'hellaswag', 'winogrande', 'ARC-Easy', 'ARC-Challenge', 'openbookqa']

# Call the function with the directory containing the JSON files
analyze_model_performance('experiment', datasets, model='768_12_4-1')
analyze_model_performance('experiment', datasets, model='768_6_4-2')
analyze_model_performance('experiment', datasets, model='768_4_4-3')