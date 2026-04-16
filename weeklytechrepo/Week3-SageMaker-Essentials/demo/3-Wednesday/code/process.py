import os
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import logging

# Configure basic logging to see outputs in CloudWatch logs when running in SageMaker
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Starting data processing ---")
    
    # ---------------------------------------------------------
    # STEP 1: Data Acquisition / Generation
    # ---------------------------------------------------------
    # In a real-world scenario, you would be loading a CSV or Parquet file 
    # from S3 via the '/opt/ml/processing/input/' directory. 
    # For this demo, we generate synthetic linear regression data on the fly.
    # The dataset has 1000 samples, 3 features, and some random noise.
    logger.info("Generating synthetic linear regression data...")
    X, y = make_regression(n_samples=1000, n_features=3, noise=0.1, random_state=42)
    
    # We combine the generated X features and y targets into a pandas DataFrame
    # to make it easy to manipulate and save to CSV files.
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    # ---------------------------------------------------------
    # STEP 2: Data Cleaning & Feature Engineering
    # ---------------------------------------------------------
    # Since the data is perfectly synthetic, no real cleaning is needed.
    # Normally, you would handle nulls, encode categorical variables, and normalize features here.

    # ---------------------------------------------------------
    # STEP 3: Data Splitting
    # ---------------------------------------------------------
    # A standard best practice is splitting data into training and validation sets
    # to evaluate model performance cleanly later. We reserve 20% for validation.
    logger.info("Splitting data into training and validation sets...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # ---------------------------------------------------------
    # STEP 4: Saving Output Artifacts
    # ---------------------------------------------------------
    # SageMaker Processing Jobs map specific container directories like 
    # '/opt/ml/processing/<output_name>' directly to S3 buckets automatically.
    # Anything written to these paths is uploaded back to the cloud upon completion.
    base_dir = "/opt/ml/processing"
    train_path = os.path.join(base_dir, "train")
    val_path = os.path.join(base_dir, "validation")
    
    try:
        # Create the directories in the container (just in case they don't exist)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        
        # Save the partitioned sets as CSVs without headers if preferred, 
        # or with headers based on the algorithm's expectations.
        train_df.to_csv(os.path.join(train_path, "train.csv"), index=False)
        val_df.to_csv(os.path.join(val_path, "validation.csv"), index=False)
        logger.info(f"Saved datasets to {base_dir}")
        
    except PermissionError:
        # LOCAL FALLBACK
        # If you run `python process.py` on your own local laptop, /opt/ml/ won't exist.
        # This fallback catches the error and writes the files locally to your project folder
        # allowing you to test the script before uploading it to SageMaker!
        logger.info(f"Writing locally since {base_dir} is inaccessible (local dev run).")
        train_df.to_csv("train.csv", index=False)
        val_df.to_csv("validation.csv", index=False)
    
    logger.info("--- Data processing complete! ---")

if __name__ == "__main__":
    main()
