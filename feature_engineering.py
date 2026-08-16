import pandas as pd
import numpy as np
import os

def main():
    data_dir = "data"
    input_file = os.path.join(data_dir, "master_data_ca.parquet")
    output_file = os.path.join(data_dir, "features_ca.parquet")
    
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    
    print("Sorting data temporally...")
    df['d_num'] = df['d'].apply(lambda x: int(x.split('_')[1])).astype(np.int16)
    df = df.sort_values(['id', 'd_num']).reset_index(drop=True)
    
    print("Creating Lag Features (Looking into the past)...")
    df['lag_28'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(28)).astype(np.float16)
    df['lag_35'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(35)).astype(np.float16)
    
    print("Creating Rolling Features (Moving averages)...")
    df['rolling_mean_7'] = df.groupby(['id'])['lag_28'].transform(lambda x: x.rolling(7).mean()).astype(np.float16)
    df['rolling_mean_28'] = df.groupby(['id'])['lag_28'].transform(lambda x: x.rolling(28).mean()).astype(np.float16)
    
    print("Creating Price Features...")
    df['price_change'] = df.groupby(['id'])['sell_price'].pct_change().astype(np.float16)
    
    print("Dropping rows with NaN values (due to lags)...")
    # FIX: We now specify exactly which columns to check for NaNs. 
    # This prevents normal, non-holiday days from being deleted.
    df = df.dropna(subset=['lag_28', 'rolling_mean_28']).reset_index(drop=True)
    
    
    print("Dropping rows with NaN values (due to lags)...")
    df = df.dropna(subset=['lag_28', 'rolling_mean_28']).reset_index(drop=True)
    
    # --- NEW IMPLEMENTATION FROM Q27.6 ---
    print("Addressing Q27.6: Removing pre-launch zero sales...")
    # This calculates a rolling sum. If the sum is 0, the item hasn't had its first sale yet.
    # We only keep rows where the cumulative sum is greater than 0.
    df = df[df.groupby(['id'])['sales'].transform(lambda x: x.cumsum()) > 0].reset_index(drop=True)
    # -------------------------------------

    
    
    print(f"Saving feature-engineered dataset to {output_file}...")
    df.to_parquet(output_file, index=False)
    print("Feature Engineering Complete! You are ready to train a model.")

if __name__ == "__main__":
    main()