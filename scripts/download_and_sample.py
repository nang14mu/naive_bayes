import io
import sys
import urllib.request
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def download_and_sample_bank_marketing(
    output_path="data/bank_marketing_1000.csv",
    sample_size=1000,
    random_state=42
):
    print("1. Đang tải tập dữ liệu Bank Marketing từ UCI...")
    url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req, timeout=60) as resp:
        zip_data = resp.read()

    print("2. Đang giải nén và đọc file bank-full.csv...")
    with zipfile.ZipFile(io.BytesIO(zip_data)) as z_outer:
        with zipfile.ZipFile(io.BytesIO(z_outer.read("bank.zip"))) as z_inner:
            with z_inner.open("bank-full.csv") as f:
                df = pd.read_csv(f, sep=";")

    target_columns = [
        "age",
        "balance",
        "job",
        "marital",
        "housing",
        "loan",
        "y"
    ]

    print(f"Tổng số bản ghi gốc: {len(df)}")
    print("Tỉ lệ nhãn gốc 'y':")
    for val, prop in df["y"].value_counts(normalize=True).items():
        print(f"  {val}: {prop:.2%}")

    print(f"\n3. Phân tầng mẫu (Stratified Sampling) lấy {sample_size} mẫu...")
    df_selected = df[target_columns].copy()

    sampled_df, _ = train_test_split(
        df_selected,
        train_size=sample_size,
        stratify=df_selected["y"],
        random_state=random_state
    )

    sampled_df = sampled_df.reset_index(drop=True)

    print(f"Số lượng mẫu sau khi trích xuất: {len(sampled_df)}")
    print("Số lượng và tỉ lệ từng nhãn:")
    counts = sampled_df["y"].value_counts()
    props = sampled_df["y"].value_counts(normalize=True)
    for val in counts.index:
        print(f"  {val}: {counts[val]} ({props[val]:.2%})")

    sampled_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n4. Đã lưu thành công vào file: {output_path}")

    return sampled_df

if __name__ == "__main__":
    download_and_sample_bank_marketing()
