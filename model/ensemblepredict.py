import os
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_squared_error, 
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import random

def round_to_half(number):
    if number < 0.5:
        return 1
    integer_part = int(number)
    decimal_part = number - integer_part
    if decimal_part < 0.5:
        return float(integer_part)  
    
    else : return float(integer_part + 1)
        
        
class EnsembleRegressor:
    def __init__(self, mlp_path='mlp_model1.pth', xgb_path='xgboost_model1.pkl', input_size=21, mlp_weight=0.2):
        """Initialize ensemble with trained models"""
        self.mlp_model = self._load_mlp_model(mlp_path, input_size)
        self.xgb_model = joblib.load(xgb_path)
        self.mlp_weight = mlp_weight
        
    def _load_mlp_model(self, path, input_size):
        """Load PyTorch MLP model"""
        class MLP(nn.Module):
          def __init__(self, input_size=21):  # Assuming 21-dimensional input
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 16)
            self.fc5 = nn.Linear(16, 16)
            self.fc4 = nn.Linear(16, 1)
            self.dropout = nn.Dropout(0.5)
        
          def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc5(x))
            x = self.dropout(x)
            x = self.fc4(x)
            return x
        
        model = MLP(input_size)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    
    def round_to_half(number):
     if number < 0.5:
        return 1
     integer_part = int(number)
     decimal_part = number - integer_part
     if decimal_part < 0.5:
        return float(integer_part)  
    
     else : return float(integer_part + 1)
            
    
    def predict(self, features):
        """Make prediction on feature array"""
        with torch.no_grad():
            mlp_pred = self.mlp_model(torch.FloatTensor(features)).numpy().flatten()
        xgb_pred = self.xgb_model.predict(features)
        raw_pred = (self.mlp_weight * mlp_pred + (1 - self.mlp_weight) * xgb_pred)
        rounded_pred = np.array([round_to_half(x) for x in raw_pred])
        return raw_pred, rounded_pred

def evaluate_and_save_results(test_dir='/data2/lhmData/ouc/AHP/pt/yangkou', 
                            output_csv='predictions.csv'):

 
    ensemble = EnsembleRegressor(mlp_weight=0.1)
    
    # Collect all test files
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.pt')]
    
    # Prepare data storage
    results = {
        'filename': [],
        'mask_type': [],  # 'original' or 'masked'
        'raw_prediction': [],
        'rounded_prediction': [],
        'true_label': [],
        'rounded_true_label': [],
        'masked_indices': [] 
    }

    for filename in test_files:
        file_path = os.path.join(test_dir, filename)
        data = torch.load(file_path)
        original_features = data['features'].numpy()
        true_label =float(data['label'])
        rounded_true_label = round_to_half(float(data['label']))
      
        features = original_features.reshape(1, -1)
        raw_pred, rounded_pred = ensemble.predict(features)
        
        results['filename'].append(filename)
        results['mask_type'].append('original')
        results['raw_prediction'].append(float(raw_pred[0]))
        results['rounded_prediction'].append(int(rounded_pred[0]))
        results['true_label'].append(true_label)
        results['rounded_true_label'].append(rounded_true_label)
        results['masked_indices'].append(None)

        masked_features = original_features.copy()
        num_to_mask = random.randint(1, min(5, len(original_features)))
        indices_to_mask = random.sample(range(len(original_features)), num_to_mask)
        
        for idx in indices_to_mask:
            masked_features[idx] = 0
        
        features = masked_features.reshape(1, -1)
        raw_pred, rounded_pred = ensemble.predict(features)
        
        results['filename'].append(filename)
        results['mask_type'].append('masked')
        results['raw_prediction'].append(float(raw_pred[0]))
        results['rounded_prediction'].append(int(rounded_pred[0]))
        results['true_label'].append(true_label)
        results['rounded_true_label'].append(rounded_true_label)
        results['masked_indices'].append(indices_to_mask)
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    original_mask = df['mask_type'] == 'original'

    print("\nOriginal Samples Metrics:")
    y_true_orig = df[original_mask]['true_label'].values
    rounded_y_true_orig = df[original_mask]['rounded_true_label'].values
    y_pred_raw_orig = df[original_mask]['raw_prediction'].values
    y_pred_round_orig = df[original_mask]['rounded_prediction'].values
    
    print("Regression Metrics:")
    print(f"MSE: {mean_squared_error(y_true_orig, y_pred_raw_orig):.4f}")
    print(f"Pearson: {pearsonr(y_true_orig, y_pred_raw_orig)[0]:.4f}")
    
    print("\nClassification Metrics:")
    print(f"Accuracy: {accuracy_score(rounded_y_true_orig, y_pred_round_orig):.4f}")
    print(f"precision: {precision_score(rounded_y_true_orig, y_pred_round_orig, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(rounded_y_true_orig, y_pred_round_orig, average='weighted', zero_division=0):.4f}")
    print(f"F1: {f1_score(rounded_y_true_orig, y_pred_round_orig, average='weighted', zero_division=0):.4f}")

    print("\nMasked Samples Metrics:")
    y_true_masked = df[~original_mask]['true_label'].values
    rounded_y_true_masked = df[original_mask]['rounded_true_label'].values
    y_pred_raw_masked = df[~original_mask]['raw_prediction'].values
    y_pred_round_masked = df[~original_mask]['rounded_prediction'].values
    
    print("Regression Metrics:")
    print(f"MSE: {mean_squared_error(y_true_masked, y_pred_raw_masked):.4f}")
    print(f"Pearson: {pearsonr(y_true_masked, y_pred_raw_masked)[0]:.4f}")
    
    print("\nClassification Metrics:")
    print(f"Accuracy: {accuracy_score(rounded_y_true_masked, y_pred_round_masked):.4f}")
    print(f"precision: {precision_score(rounded_y_true_masked, y_pred_round_masked, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(rounded_y_true_masked, y_pred_round_masked, average='weighted', zero_division=0):.4f}")
    print(f"F1: {f1_score(rounded_y_true_masked, y_pred_round_masked, average='weighted', zero_division=0):.4f}")
    
    return df

if __name__ == "__main__":
    results_df = evaluate_and_save_results()
    