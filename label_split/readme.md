# Video Label Splitting Tools

This repository provides scripts to help organize and label colonoscopy videos using accompanying Excel files.

---

## 📁 Split Video Labels

If you have downloaded our open dataset, you will find video files and Excel label files.

To split the videos based on their labels:

1. Run `split_video_excel.py`  
   → This script organizes all video and Excel files into a `video_excel` folder.

2. Run `split_video_label.py`  
   → This script splits the videos according to the corresponding label information in the Excel files.

---

## 📁 Split OpenDataset Labels

To split and organize OpenDataset labels:

1. Download all OpenDataset label files and the corresponding videos.
2. Make sure the dataset folder contains both the videos and the Excel label file.
3. Set the folder path in `split_opendata_label.py` and run the script.

→ This will automatically organize the OpenDataset videos based on their label information.

---

## 📌 Example

```bash
# Step 1: Organize video and Excel files
python split_video_excel.py --input_folder ./raw_data

# Step 2: Split by video label
python split_video_label.py --input_folder ./video_excel

# Step 3: For OpenDataset
python split_opendata_label.py --input_folder ./opendata

