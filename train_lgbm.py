import pandas as pd
import lightgbm as lgb
import os
from sklearn.metrics import mean_squared_error
import numpy as np

def main():
    data_dir = "data"
    input_file = os.path.join(data_dir, "features_ca.parquet")
    
    print("Loading feature-engineered data (this takes a moment)...")
    df = pd.read_parquet(input_file)
    
    print("Converting all text columns to 'category' dtype for LightGBM...")
    # Foolproof method: grab everything that is NOT a number and make it a category
    cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns
    for col in cat_cols:
        df[col] = df[col].astype('category')
            
    # We use the last 28 days for validation, and everything prior for training.
    val_start_day = df['d_num'].max() - 27
    
    print(f"Splitting data into Train (< day {val_start_day}) and Validation (>= day {val_start_day})...")
    
    # Features to exclude from training
    exclude_cols = ['id', 'd', 'date', 'sales', 'd_num', 'wm_yr_wk'] 
    features = [col for col in df.columns if col not in exclude_cols]
    
    train_data = df[df['d_num'] < val_start_day]
    val_data = df[df['d_num'] >= val_start_day]
    
    X_train, y_train = train_data[features], train_data['sales']
    X_val, y_val = val_data[features], val_data['sales']
    
    print(f"Training on {len(X_train):,} rows. Validating on {len(X_val):,} rows.")
    print(f"Features used: {len(features)}")
    
    # Define LightGBM parameters
    params = {
        'objective': 'tweedie',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'n_estimators': 150,
        'random_state': 42,
        'n_jobs': -1 
    }
    
    print("\nTraining LightGBM baseline model...")
    model = lgb.LGBMRegressor(**params)
    
    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )
    
    print("\nGenerating predictions on validation set...")
    preds = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"Validation RMSE: {rmse:.4f}")
    
    # Save the trained model
    model_path = os.path.join(data_dir, "lgbm_baseline.txt")
    model.booster_.save_model(model_path)
    print(f"\nModel successfully saved to {model_path}")

if __name__ == "__main__":
    main()
    
