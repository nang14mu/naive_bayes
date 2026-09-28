import sys
import pandas as pd
from sklearn.model_selection import train_test_split

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def split_bank_marketing(
    input_path="data/bank_marketing_1000.csv",
    test_size=0.2,
    random_state=42
):
    print(f"1. Đọc dữ liệu từ {input_path}...")
    df = pd.read_csv(input_path)
    total_len = len(df)
    print(f"Tổng số mẫu: {total_len}")
    
    print("\n2. Phân tách tập Train (80%) và Test (20%) có bảo toàn tỉ lệ nhãn (Stratified)...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["y"],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"- Tập Train: {len(train_df)} mẫu ({len(train_df)/total_len:.0%})")
    for val in train_df["y"].value_counts().index:
        count = train_df["y"].value_counts()[val]
        prop = train_df["y"].value_counts(normalize=True)[val]
        print(f"    + {val}: {count} ({prop:.2%})")

    print(f"- Tập Test: {len(test_df)} mẫu ({len(test_df)/total_len:.0%})")
    for val in test_df["y"].value_counts().index:
        count = test_df["y"].value_counts()[val]
        prop = test_df["y"].value_counts(normalize=True)[val]
        print(f"    + {val}: {count} ({prop:.2%})")

    # Lưu vào cả data/train/ và data/
    train_df.to_csv("data/train/train.csv", index=False, encoding="utf-8")
    train_df.to_csv("data/train.csv", index=False, encoding="utf-8")
    
    test_df.to_csv("data/test/test.csv", index=False, encoding="utf-8")
    test_df.to_csv("data/test.csv", index=False, encoding="utf-8")

    print("\n3. Đã lưu thành công:")
    print("  - Train: data/train/train.csv (và data/train.csv)")
    print("  - Test:  data/test/test.csv (và data/test.csv)")

    return train_df, test_df

if __name__ == "__main__":
    split_bank_marketing()
