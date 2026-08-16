import pandas as pd
import lightgbm as lgb
import os
import numpy as np

def main():
    data_dir = "data"
    input_file = os.path.join(data_dir, "features_ca.parquet")
    
    print("Loading feature-engineered data...")
    df = pd.read_parquet(input_file)
    
    print("Converting text columns to 'category' dtype...")
    cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns
    for col in cat_cols:
        df[col] = df[col].astype('category')
            
    val_start_day = df['d_num'].max() - 27
    
    exclude_cols = ['id', 'd', 'date', 'sales', 'd_num', 'wm_yr_wk'] 
    features = [col for col in df.columns if col not in exclude_cols]
    
    train_data = df[df['d_num'] < val_start_day]
    val_data = df[df['d_num'] >= val_start_day].copy() # Copy to store predictions safely
    
    X_train, y_train = train_data[features], train_data['sales']
    X_val = val_data[features]
    
    # We want to predict the lower bound (5%), median (50%), and upper bound (95%)
    quantiles = [0.05, 0.50, 0.95]
    
    # Dictionary to store the models
    models = {}
    
    print("\n--- Starting Probabilistic Training ---")
    
    for alpha in quantiles:
        print(f"\nTraining model for Quantile: {alpha}")
        
        # Notice the objective is now 'quantile' and we pass 'alpha'
        params = {
            'objective': 'quantile',
            'alpha': alpha,
            'metric': 'quantile',
            'learning_rate': 0.05,
            'n_estimators': 100, # Keeping it low for speed
            'random_state': 42,
            'n_jobs': -1 
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        # Predict and store in our validation dataframe
        val_data[f'pred_q{int(alpha*100)}'] = model.predict(X_val)
        
        # Save each model
        model_path = os.path.join(data_dir, f"lgbm_model_q{int(alpha*100)}.txt")
        model.booster_.save_model(model_path)
        
    print("\n--- Training Complete ---")
    
    # Let's preview what probabilistic predictions look like for the first 5 rows
    preview = val_data[['id', 'd', 'sales', 'pred_q5', 'pred_q50', 'pred_q95']].head()
    print("\nSample Probabilistic Forecasts:")
    print(preview.to_string(index=False))
    
    # Save the predictions to analyze later
    output_file = os.path.join(data_dir, "probabilistic_predictions.parquet")
    val_data.to_parquet(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")

if __name__ == "__main__":
    main()