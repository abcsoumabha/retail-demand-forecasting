import pandas as pd
import os

def main():
    data_dir = "data"
    input_file = os.path.join(data_dir, "probabilistic_predictions.parquet")
    output_file = os.path.join(data_dir, "reconciled_forecasts.parquet")
    
    print("Loading base forecasts...")
    df = pd.read_parquet(input_file)
    
    # IMPLEMENTATION Q27.8: Seasonal Naive Baseline (7-day lag for weekly seasonality)
    print("Calculating Seasonal Naive baseline (Q27.8)...")
    df = df.sort_values(['store_id', 'item_id', 'd_num'])
    df['snaive'] = df.groupby(['store_id', 'item_id'])['sales'].shift(7)
    
    # IMPLEMENTATION Q27.16: Bottom-Up Hierarchical Reconciliation
    print("Performing Bottom-Up Hierarchical Reconciliation (Q27.16)...")
    hierarchy_list = []
    
    # Level 1: Bottom Level (Item)
    df_item = df.copy()
    df_item['Level'] = 'Item'
    df_item['Entity'] = df_item['item_id']
    hierarchy_list.append(df_item[['d_num', 'Level', 'Entity', 'sales', 'pred_q5', 'pred_q50', 'pred_q95', 'snaive']])
    
    # Level 2: Middle Level (Department)
    df_dept = df.groupby(['d_num', 'dept_id'])[['sales', 'pred_q5', 'pred_q50', 'pred_q95', 'snaive']].sum().reset_index()
    df_dept['Level'] = 'Department'
    df_dept['Entity'] = df_dept['dept_id']
    hierarchy_list.append(df_dept[['d_num', 'Level', 'Entity', 'sales', 'pred_q5', 'pred_q50', 'pred_q95', 'snaive']])
    
    # Level 3: Top Level (Store)
    df_store = df.groupby(['d_num', 'store_id'])[['sales', 'pred_q5', 'pred_q50', 'pred_q95', 'snaive']].sum().reset_index()
    df_store['Level'] = 'Store'
    df_store['Entity'] = df_store['store_id']
    hierarchy_list.append(df_store[['d_num', 'Level', 'Entity', 'sales', 'pred_q5', 'pred_q50', 'pred_q95', 'snaive']])
    
    # Combine all hierarchical levels into one dataset
    reconciled_df = pd.concat(hierarchy_list, ignore_index=True)
    
    # Drop rows where Seasonal Naive is NaN (the first 7 days of validation)
    reconciled_df = reconciled_df.dropna(subset=['snaive'])
    
    reconciled_df.to_parquet(output_file, index=False)
    print(f"Hierarchical forecasts successfully saved to {output_file}")

if __name__ == "__main__":
    main()
    
