import pandas as pd
import os

def ensemble():
    directory_path = 'results'
    file_list = [file for file in os.listdir(directory_path) if file.endswith('.csv')]
    
    df_list = []

    for file in file_list:
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)

    sum_df = sum(df.iloc[:, 1:] for df in df_list) / len(df_list)
    
    ensemble_df = pd.concat([df_list[0].iloc[:, [0]], sum_df], axis=1)
    ensemble_df.to_csv('results/ensemble_results.csv', index=False)