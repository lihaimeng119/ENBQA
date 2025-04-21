import pandas as pd
import numpy as np
import re
import torch

data = pd.read_csv('data/traindata.csv', header=None)

id_column = data.iloc[1:, 0] 
features = data.iloc[1:, 1:]

weights_data = pd.read_csv('weight.csv', header=None)
weights = weights_data.iloc[1:, 1].values.astype(float)

weighted_scores_data = pd.read_csv('data/trainweightscore.csv', header=None)  
weighted_scores = weighted_scores_data.iloc[1:, 22].values.astype(float)  
weighted_ids = weighted_scores_data.iloc[1:, 0].values[:]  

mapping_data = pd.read_excel('datagenerate.xlsx', header=None)
feature_names = data.iloc[0, 1:].values 

'''def round_to_half(number):
    integer_part = int(number)
    decimal_part = number - integer_part
    
    if decimal_part < 0.25:
        return float(integer_part)  
    elif decimal_part < 0.75:
        return float(integer_part + 0.5)  
    else:
        return float(integer_part + 1) '''
        
def round_to_half(number):
    if number < 0.5:
        return 1
    integer_part = int(number)
    decimal_part = number - integer_part
    if decimal_part < 0.5:
        return float(integer_part)  
    else:
        return float(integer_part + 1)
        

for row_index in range(features.shape[0]):
    idx = id_column.iloc[row_index]
    
    if idx not in weighted_ids:
        print(idx)
        continue  
    
    feature_vector = []  
    
    for col_index in range(features.shape[1]): 
        feature_name = feature_names[col_index] 
        
        value = features.iloc[row_index, col_index]
        
        if isinstance(value, str) and re.search(r'\d', value):
            value = float(value)

        if isinstance(value, (int, float)): 
            feature_vector.append(round(float(value), 2))
        else:
            matching_columns = mapping_data.columns[mapping_data.isin([value]).any(axis=0)] 
            
            if not matching_columns.empty:
                score = mapping_data[matching_columns[0]].iloc[0] 
                feature_vector.append(score)
            else:
                feature_vector.append(None) 
    
    current_feature_vector = pd.Series(feature_vector, index=feature_names)

    current_feature_vector.fillna(0, inplace=True)  

    current_feature_vector_values = current_feature_vector.values.astype(float)
    
    total = current_feature_vector_values.sum() 
    print(f'Total for index {idx}: {total}')

    if total != 0:
        current_feature_vector_values = current_feature_vector_values / total * 1000
    else:
        current_feature_vector_values = current_feature_vector_values * 0
    
    
    weighted_vector = current_feature_vector_values * weights[:len(current_feature_vector_values)]
    weighted_vector = np.round(weighted_vector, 3)

    label_index = np.where(weighted_ids == idx)[0][0] 
    label = weighted_scores[label_index]

   
    tensor = torch.tensor(weighted_vector, dtype=torch.float32) 
    filename = f'pt/train2/{idx}.pt' 
    
    torch.save({'features': tensor, 'label': label}, filename)  
    #print(f'Saved {filename} label: {label}')