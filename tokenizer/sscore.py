import pandas as pd


weights_df = pd.read_csv('weight.csv', header=None)  
weights = dict(zip(weights_df[0], weights_df[1]))  


scores_df = pd.read_csv('data/2.csv')

score_columns = scores_df.columns[1:] 
weighted_scores = []


for index, row in scores_df.iterrows():
    total_score = 0
    for column in score_columns:
        factor_score = row[column]
        try:
            factor_score = float(factor_score) 
        except ValueError:
            
            continue  
        
        if column in weights:
            weight = weights[column]
            weight = float(weight)
            
            total_score += factor_score * weight  
    if(total_score<0.5):
            print(total_score)
            
    weighted_scores.append(total_score)
    

scores_df['加权得分'] = weighted_scores


ranges = {
    '0-1': (0, 0.5),
    '0-1.': (0.5, 1),
    '1-2': (1, 1.5),
    '1-2.': (1.5, 2),
    '2-3': (2, 2.5),
    '2-3.': (2.5, 3),
    '3-4': (3, 3.5),
    '3-4.': (3.5, 4),
    '4-5': (4, 4.5),
    '4-5.': (4.5, 5)
}

train_data = []

for range_name, (lower, upper) in ranges.items():
    
    filtered_data = scores_df[(scores_df['加权得分'] >= lower) & (scores_df['加权得分'] < upper)]
    
    selected_data = filtered_data.sample(n=min(2000, len(filtered_data)), random_state=1)  # 随机选择2000条
    train_data.append(selected_data)

train_df = pd.concat(train_data, ignore_index=True)

train_df.to_csv('data/testshilaoren.csv', index=False, encoding='utf-8-sig')