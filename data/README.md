#  Data Directory

This folder is used to store the raw and processed datasets for the Dubai Real Estate Analytics project. 

**Note to Reviewers:** The raw data files are **not committed** to this repository due to GitHub's file size limits (the raw CSV is 241 MB and the Parquet file is 23 MB).

### 📥 How to get the data
To reproduce this project, you need to download the raw dataset manually:
1. Go to the [Hugging Face Dataset](https://huggingface.co/datasets/dubairealestatedata/dubai-real-estate-sales-transactions/tree/main).
2. Download the `dld_sales_transactions.parquet` file.
3. Place the file directly inside the `data/raw/` folder.

### ⚙️ How to process the data
Once the raw file is in place, run the cleaning pipeline from the root directory:
```bash
python src/cleaning.py