import pandas as pd
import config as cg

def load_raw_data():
    return pd.read_csv(cg.DATA_PATH)

def clean_data(df):
    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Drop leakage
    df = df.drop(columns=["customerID"])

    # Binary encoding
    binary_cols = ["Partner", "Dependents", "PaperlessBilling", "Churn"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    return df

if __name__ == "__main__":
    df = load_raw_data()
    df = clean_data(df)

    print(df.info())
    print(df.head())
    print(df["Churn"].value_counts())
    
    df.to_csv(cg.PROCESSED_DATA_PATH, index=False)


