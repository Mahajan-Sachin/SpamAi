import pandas as pd
import os

def load_original_data():
    return pd.read_csv("../data/spam.csv", encoding="latin-1")[["v1", "v2"]].rename(
        columns={"v1": "label", "v2": "message"}
    )

def load_new_data():
    if not os.path.exists("../new_training_samples.csv"):
        print("No new training samples found.")
        return pd.DataFrame(columns=["message", "label"])
    
    df = pd.read_csv("../new_training_samples.csv")
    df = df[["message", "label"]]
    return df

def merge_datasets():
    df_original = load_original_data()
    df_new = load_new_data()

    # Ensure column names match original dataset
    df_new = df_new.rename(columns={"label": "label", "message": "message"})

    combined = pd.concat([df_original, df_new], ignore_index=True)

    print(f"Original samples: {len(df_original)}")
    print(f"New samples: {len(df_new)}")
    print(f"Total combined: {len(combined)}")

    return combined

if __name__ == "__main__":
    merged_df = merge_datasets()
    merged_df.to_csv("../retraining/combined_dataset.csv", index=False)
    print("Combined dataset saved to retraining/combined_dataset.csv")
