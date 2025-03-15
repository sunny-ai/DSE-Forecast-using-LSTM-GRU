import os
import pandas as pd

folder_path = 'data'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

dataframes = []

for csv_file in csv_files:
    company_name = os.path.splitext(csv_file)[0]  # Remove .csv
    company_name = company_name.replace('_data', '')  # Remove _data suffix
    
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)
    
    df['Company'] = company_name
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
columns = combined_df.columns.tolist()

if 'Company' in columns:
    columns.remove('Company')

columns.insert(1, 'Company')
combined_df = combined_df[columns]
combined_df.to_csv('combined_data.csv', index=False)

print("All CSV files have been combined into 'combined_data.csv'")