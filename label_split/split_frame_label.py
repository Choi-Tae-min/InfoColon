import os
import re
import csv
import shutil
import pandas as pd

# ===== Setting =====
ROOT_DIR = "a"                 # Example: a/CNUH_V0001/CNUH_V0001_frames + CNUH_V0001_Label.csv
OUTPUT_ROOT = "output"         # Root folder for results (created if missing)
FRAME_DIR_SUFFIX = "_frames"   # Frame folder suffix
CSV_SUFFIX = "_Label.csv"      # Label CSV suffix
ALLOW_MULTI_LABEL = True       # If multiple labels, copy into all label folders
LABEL_SEP_PATTERN = r"[,\|;]"  # Allowed label separators
LOG_MISMATCH = True            # Save mismatch logs
# =================

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def six_digits_from_name(name: str):
     """Extract the last 6-digit number from filename (e.g., CNUH_V0001_000062.png -> '000062')."""
    m = re.search(r"(\d{6})\.png$", str(name).strip(), flags=re.IGNORECASE)
    return m.group(1) if m else None

def list_frames_sorted(frames_dir: str):
    """List .png frames in numeric order based on last 6 digits."""
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith(".png")]
    items = []
    for f in files:
        key = six_digits_from_name(f)
        if key is not None:
            items.append((int(key), f))
    items.sort(key=lambda x: x[0])
    return [fname for _, fname in items]

def read_csv_rows(csv_path: str):
    """Read CSV rows in order, returning (Filename, Label, split/None)."""
    df = pd.read_csv(csv_path)
    # 필수 컬럼 체크
    if "Filename" not in df.columns or "Label" not in df.columns:
        raise ValueError(f"CSV missing 'Filename' or 'Label' column: {csv_path}")
    has_split = "train_val_test" in df.columns

    rows = []
    for _, row in df.iterrows():
        fn = str(row["Filename"]).strip()
        lb = str(row["Label"]).strip()
        sp = str(row["train_val_test"]).strip() if has_split else None
        rows.append((fn, lb, sp))
    return rows

def split_labels(label_str: str):
    """Split label string into list using separators."""
    if not label_str:
        return []
    parts = re.split(LABEL_SEP_PATTERN, label_str)
    return [p.strip() for p in parts if p.strip()]

def match_and_copy(video_folder: str, frames_dir: str, csv_path: str, out_root: str):
    """Match frames and CSV rows 1:1, copy into OUTPUT_ROOT/[split]/[label]."""
    base = os.path.splitext(os.path.basename(csv_path))[0].replace(CSV_SUFFIX.replace(".csv",""), "")

    frames = list_frames_sorted(frames_dir)
    rows = read_csv_rows(csv_path)

    n_frames = len(frames)
    n_rows = len(rows)
    n = min(n_frames, n_rows)

    if n == 0:
        print(f"⚠️  No items to match: {video_folder}")
        return


    mismatches = []
    if LOG_MISMATCH and (n_frames != n_rows):
        mismatches.append(f"[COUNT] frames={n_frames}, csv_rows={n_rows}, matched={n}")


    copied = 0
    for i in range(n):
        frame_name = frames[i]                      
        csv_filename, label_str, split = rows[i]   
        src_path = os.path.join(frames_dir, frame_name)

        labels = split_labels(label_str) if ALLOW_MULTI_LABEL else [label_str]


        if split and isinstance(split, str) and split != "" and split.lower() != "nan":
            for lab in labels if labels else ["_nolabel"]:
                dst_dir = os.path.join(out_root, split, lab)
                ensure_dir(dst_dir)
                dst_path = os.path.join(dst_dir, frame_name)
                shutil.copy2(src_path, dst_path)
                copied += 1
        else:
            for lab in labels if labels else ["_nolabel"]:
                dst_dir = os.path.join(out_root, lab)
                ensure_dir(dst_dir)
                dst_path = os.path.join(dst_dir, frame_name)
                shutil.copy2(src_path, dst_path)
                copied += 1


        if LOG_MISMATCH:
            csv_key = six_digits_from_name(csv_filename)
            frm_key = six_digits_from_name(frame_name)
            if csv_key is None or frm_key is None or csv_key != frm_key:
                mismatches.append(f"[ROW {i}] csv='{csv_filename}' vs frame='{frame_name}'")

    if LOG_MISMATCH:
        if n_frames > n_rows:
            mismatches.append(f"[EXTRA] frames has {n_frames-n_rows} more items (ignored).")
        elif n_rows > n_frames:
            mismatches.append(f"[MISSING] csv has {n_rows-n_frames} more rows (no files).")

    if LOG_MISMATCH and mismatches:
        log_path = os.path.join(video_folder, "match_report.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(mismatches))
        print(f"📝 Match report saved: {log_path}")

    print(f"✅ Copy done: {copied} files  →  {out_root}")

def main():
    ensure_dir(OUTPUT_ROOT)

    for entry in sorted(os.listdir(ROOT_DIR)):
        video_folder = os.path.join(ROOT_DIR, entry)
        if not os.path.isdir(video_folder):
            continue


        frames_dirs = [os.path.join(video_folder, d)
                       for d in os.listdir(video_folder)
                       if d.endswith(FRAME_DIR_SUFFIX) and os.path.isdir(os.path.join(video_folder, d))]

        csv_candidates = [os.path.join(video_folder, f)
                          for f in os.listdir(video_folder)
                          if f.endswith(CSV_SUFFIX) and os.path.isfile(os.path.join(video_folder, f))]

        if not frames_dirs:
            print(f"ℹ️  No frame folder: {video_folder}")
            continue
        if not csv_candidates:
            print(f"ℹ️ No CSV: {video_folder}")
            continue

        frames_dir = sorted(frames_dirs)[0]
        csv_path = sorted(csv_candidates)[0]

        print(f"▶ processing: {video_folder}")
        print(f"   frames: {frames_dir}")
        print(f"   csv   : {csv_path}")

        out_root = OUTPUT_ROOT
        ensure_dir(out_root)

        match_and_copy(video_folder, frames_dir, csv_path, out_root)

if __name__ == "__main__":
    main()

