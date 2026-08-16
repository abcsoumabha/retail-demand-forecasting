import pandas as pd
import numpy as np
import os

def reduce_mem_usage(df):
    """Iterate through all columns and downcast numeric types to save memory."""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"  -> Initial memory: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Check if the column is actually numeric before trying to compress it
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Downcast Integers
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            # Downcast Floats
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    
        # If it is text/string/object, convert to category
        elif col_type == 'object' or str(col_type) == 'string':
            df[col] = df[col].astype('category')
            
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"  -> Final memory: {end_mem:.2f} MB (Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%)")
    return df

def main():
    data_dir = "data"
    
    # 1. Load Calendar and Prices
    print("Loading Calendar...")
    calendar = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
    calendar = reduce_mem_usage(calendar)
    
    print("\nLoading Sell Prices...")
    prices = pd.read_csv(os.path.join(data_dir, "sell_prices.csv"))
    prices = reduce_mem_usage(prices)
    
    # 2. Load Sales and filter to California instantly to save RAM
    print("\nLoading Sales (filtering to CA to prevent RAM crashes)...")
    sales = pd.read_csv(os.path.join(data_dir, "sales_train_validation.csv"))
    sales = sales[sales['state_id'] == 'CA'].reset_index(drop=True)
    sales = reduce_mem_usage(sales)
    
    # 3. Melt from Wide to Long Format
    print("\nMelting sales data to long format (this may take a minute)...")
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_long = pd.melt(sales, id_vars=id_vars, var_name='d', value_name='sales')
    sales_long = reduce_mem_usage(sales_long) 
    
    # 4. Merge Data together
    print("\nMerging data with calendar...")
    master_df = pd.merge(sales_long, calendar, on='d', how='left')
    
    print("Merging data with sell prices...")
    master_df = pd.merge(master_df, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    
    print("\nFinal memory optimization on merged dataset...")
    master_df = reduce_mem_usage(master_df)
    
    # 5. Save as a Parquet file
    output_file = os.path.join(data_dir, "master_data_ca.parquet")
    print(f"\nSaving ML-ready dataset to {output_file}...")
    master_df.to_parquet(output_file, index=False)
    
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()