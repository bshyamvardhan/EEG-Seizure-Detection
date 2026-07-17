import os
import numpy as np
import pandas as pd

# =====================================================
# CONFIGURATION
# =====================================================

# 👉 CHANGE THIS ONLY IF YOUR PROJECT PATH CHANGES
ROOT_DIR = "/Users/gunupatibhuvan/MyFiles/MiniProject"

DATASET_PATH = os.path.join(ROOT_DIR, "Base_Dataset")
OUTPUT_DIR = os.path.join(ROOT_DIR, "Feature_Extraction/csv_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LABEL MAPPING
# =====================================================
FOLDERS = {
    "S": "ictal",
    "F": "inter_ictal",
    "N": "inter_ictal",
    "Z": "healthy",
    "O": "healthy"
}

# =====================================================
# PARAMETERS (adjust if needed)
# =====================================================
DT = 1 / 173.61  # sampling interval

# =====================================================
# PROCESS ONE FOLDER
# =====================================================
def process_folder(folder_path, label):
    rows = []

    if not os.path.exists(folder_path):
        print(f"⚠ Folder not found: {folder_path}")
        return pd.DataFrame(columns=["time_sec", "amplitude", "class"])

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)

            try:
                signal = np.loadtxt(file_path)
                signal = signal.flatten()

                time_axis = np.arange(len(signal)) * DT

                for t, amp in zip(time_axis, signal):
                    rows.append([t, amp, label])

            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")

    return pd.DataFrame(rows, columns=["time_sec", "amplitude", "class"])

# =====================================================
# MAIN EXECUTION
# =====================================================
def main():

    final_data = {
        "ictal": [],
        "inter_ictal": [],
        "healthy": []
    }

    # -----------------------------
    # PROCESS ALL FOLDERS
    # -----------------------------
    for folder_name, label in FOLDERS.items():
        folder_path = os.path.join(DATASET_PATH, folder_name)

        print(f"Processing: {folder_path} -> {label}")

        df = process_folder(folder_path, label)
        final_data[label].append(df)

    # -----------------------------
    # SAVE FINAL CSV FILES
    # -----------------------------
    for label, dfs in final_data.items():

        if len(dfs) == 0:
            print(f"⚠ No data for {label}")
            continue

        final_df = pd.concat(dfs, ignore_index=True)

        output_file = os.path.join(OUTPUT_DIR, f"{label}.csv")
        final_df.to_csv(output_file, index=False)

        print(f"✅ Saved: {output_file} | Shape: {final_df.shape}")

    print("\n🎉 Dataset creation completed successfully!")

# =====================================================
# RUN SCRIPT
# =====================================================
if __name__ == "__main__":
    main()