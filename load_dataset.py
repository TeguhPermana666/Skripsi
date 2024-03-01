import pandas as pd
def load_dataset_with_label(csv_path, result, target_label):
    df_fashion = pd.read_csv(csv_path, index_col=0)
    filtered_df = df_fashion[df_fashion['label'] == target_label]
    result.append((filtered_df['label'].values, filtered_df.drop('label', axis=1).values))