import pandas as pd
import os
import numpy as np

def main():
    data_dir = "data"
    input_file = os.path.join(data_dir, "reconciled_forecasts.parquet")
    output_file = os.path.join(data_dir, "demo_forecasts.parquet")
    
    print(f"Loading massive dataset: {input_file}...")
    df = pd.read_parquet(input_file)
    
    # 1. Keep ALL Store and Department level forecasts
    top_levels = df[df['Level'].isin(['Store', 'Department'])]
    
    # 2. Keep only a sample of 200 Items to drastically shrink the file
    item_df = df[df['Level'] == 'Item']
    sample_items = item_df['Entity'].drop_duplicates().head(200)
    item_demo = item_df[item_df['Entity'].isin(sample_items)]
    
    # 3. Combine them back together
    demo_df = pd.concat([top_levels, item_demo], ignore_index=True)
    
    # 4. Downcast floats from 64-bit to 32-bit to save even more space
    float_cols = ['sales', 'pred_q5', 'pred_q50', 'pred_q95', 'snaive']
    for col in float_cols:
        demo_df[col] = demo_df[col].astype(np.float32)
        
    demo_df.to_parquet(output_file, index=False)
    
    original_size = os.path.getsize(input_file) / (1024 * 1024)
    new_size = os.path.getsize(output_file) / (1024 * 1024)
    
    print(f"Success! Original size: {original_size:.2f} MB")
    print(f"Demo file size: {new_size:.2f} MB")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()