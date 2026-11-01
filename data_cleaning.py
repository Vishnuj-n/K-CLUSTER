import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_and_prepare_data(input_csv_path="dataset/Mall_Customers.csv", output_dir="dataset"):
    # 1. Load data
    df = pd.read_csv(input_csv_path)
    print("--- Initial Data Summary ---")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}\n")
    print(f"Duplicates: {df.duplicated().sum()}\n")

    # 2. Basic Cleaning
    # Standardize column names (strip whitespace, lowercase, replace spaces/special chars)
    df.columns = [c.strip() for c in df.columns]
    
    # Remove duplicates if any
    df = df.drop_duplicates()
    
    # Handle missing values if any
    df = df.dropna()

    # Drop identifier column not relevant for clustering distances
    if "CustomerID" in df.columns:
        df_features = df.drop(columns=["CustomerID"])
    else:
        df_features = df.copy()

    # Encode categorical variables: Gender
    if "Gender" in df_features.columns:
        # Binary encoding: Female -> 0, Male -> 1
        df_features["Gender"] = df_features["Gender"].map({"Female": 0, "Male": 1})

    # Save cleaned unscaled dataset (useful for human interpretation after clustering)
    cleaned_unscaled_path = os.path.join(output_dir, "cleaned_customers.csv")
    df_features.to_csv(cleaned_unscaled_path, index=False)
    print(f"Cleaned (unscaled) dataset saved to: {cleaned_unscaled_path}")

    # 3. Feature Scaling (Essential for distance-based algorithms like K-Means)
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_features)
    df_scaled = pd.DataFrame(scaled_array, columns=df_features.columns)

    # Save standardized dataset (ready to be fed directly into KMeans)
    cleaned_scaled_path = os.path.join(output_dir, "scaled_customers_kmeans.csv")
    df_scaled.to_csv(cleaned_scaled_path, index=False)
    print(f"Scaled dataset (K-Means ready) saved to: {cleaned_scaled_path}")

    # 4. Optional / Common 2D subset for easy visualization: Annual Income & Spending Score
    subset_cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    if all(col in df_features.columns for col in subset_cols):
        scaler_2d = StandardScaler()
        df_2d_scaled = pd.DataFrame(
            scaler_2d.fit_transform(df_features[subset_cols]),
            columns=subset_cols
        )
        subset_path = os.path.join(output_dir, "scaled_income_spending_kmeans.csv")
        df_2d_scaled.to_csv(subset_path, index=False)
        print(f"2D subset (Income & Spending) saved to: {subset_path}")

    print("\n--- Preview of Scaled Data ---")
    print(df_scaled.head())
    print("\n--- Summary Statistics of Scaled Data ---")
    print(df_scaled.describe().round(2))

if __name__ == "__main__":
    clean_and_prepare_data()
