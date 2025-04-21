import pandas as pd
import random


data_df = pd.read_csv('model/test.csv')


scores = []


df = pd.read_excel('datagenerate.xlsx')


for index, row in data_df.iterrows():
    score_row = []
    for idx in range(len(df)):
        factor = df.iloc[idx, 0] 
        scores_list = df.iloc[idx, 1:].tolist()

        if factor in row:
            score_index = random.randint(0, len(scores_list) - 1)
            score = scores_list[score_index]
            score_row.append(score_index + 1) 
        else:
            score_row.append(0)  
            if scores_list:
                score_row_value = scores_list[0] 
                score_row[-1] = score_row_value 

    scores.append(score_row)


score_df = pd.DataFrame(scores)
score_df.columns = list(range(1, len(data_df.columns)))  
score_df.insert(0, '序号', range(1, len(data_df) + 1))  
score_df.columns = ['序号'] + list(range(1, len(data_df.columns))) 

score_df.to_csv('data/2.csv', index=False, encoding='utf-8-sig')

print("得分文件已保存为 'data/testscore.csv'")