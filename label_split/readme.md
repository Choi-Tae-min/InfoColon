# Video Label Splitting Tools

This repository provides scripts to help organize and label colonoscopy videos using accompanying Excel files.

---

## 📁 Split Video Labels

If you have downloaded our open dataset, you will find video files and Excel label files.

To split the videos based on their labels:

1. Run `videoframe.py`  
   → This script extracts frames from each video and organizes them into individual frame folders.

2. Run `split_frame_label.py`  
   → This script reorganizes the extracted frames into folders according to the label information in the Excel files.

---

## 📁 Split OpenDataset Labels

To split and organize OpenDataset labels:

1. Download all OpenDataset label files and the corresponding videos.
2. Make sure the dataset folder contains both the videos and the Excel label file.
3. Set the folder path in `split_opendata_label.py` and run the script.

→ This will automatically organize the OpenDataset videos based on their label information.

---

## 📌 Example


# Step 1: Extract video to frame and save each frame folder

```bash
python videoframe.py --root ./where_you_save_our_hospital_data_folder(SNUH_Colonscopy, CNUH_Colonoscopy) --mode scaled_time --start-sec 0 --save-images 1
'''

# Step 2: Split by video label

```bash
python split_frame_label.py --root ./where_you_save_our_hospital_video_data_folder
'''
# Step 3: For OpenDataset
Nerthus

```bash
python opendataset_split.py --excel-glob "\{Opendataset_Excel_Path}\*_nerthus_nerthus-dataset-frames_output*.csv" --root "\data\" --dest "\data\output" --path-col Path --name-col Filename --label-col Label --strip-output-mode always --strip-keywords nerthus
'''

Endomapper

```bash
python opendataset_split.py --excel-glob "\{Opendataset_Excel_Path}\*_endomapper_*.csv" --root "\data\" --dest "\data\output" --path-col Path --name-col Filename --label-col Label --allow-substring
'''

Hyper-kvasir

```bash
python opendataset_split.py --excel-glob "\{Opendataset_Excel_Path}\*_hyper-kvasir_*.csv" --root "\data\" --dest "\data\output" --path-col Path --name-col Filename --label-col Label --allow-substring
'''
