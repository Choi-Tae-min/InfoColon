import os, math, cv2, argparse, csv
from typing import List, Optional
import pandas as pd

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg")

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def secs_from_video(total_frames: int, fps: float, start_sec=1, step_sec=1) -> List[int]:
    """Generate integer second list from video metadata."""
    if fps <= 0 or total_frames <= 0:
        return []
    max_sec = int(math.floor((total_frames - 1) / fps))
    return list(range(start_sec, max_sec + 1, step_sec))

def secs_from_csv_len(csv_path: str, start_sec=1, step_sec=1) -> List[int]:
    """Force integer second list based on CSV length (Option A)."""
    df = pd.read_csv(csv_path)
    n = len(df)
    return [start_sec + i * step_sec for i in range(n)]

def ideal_ms_of(sec: int, offset_ms: float, scale: float) -> float:
    """Ideal timestamp (ms) for the target second. Adjusted with scale and offset."""
    return sec * 1000.0 * scale + offset_ms

def find_label_csv_for_video(video_path: str, csv_suffix="_Label.csv") -> Optional[str]:
    folder = os.path.dirname(video_path)
    base = os.path.splitext(os.path.basename(video_path))[0]
    cand = os.path.join(folder, base + csv_suffix)
    return cand if os.path.isfile(cand) else None

def extract_one(
    video_path: str,
    start_sec: int = 1,
    step_sec: int = 1,
    mode: str = "video",            # "video"=default, "csv_len"=Option A, "scaled_time"=Option B
    csv_for_len: Optional[str] = None,
    offset_ms: float = 0.0,         # Common offset (ms)
    scale: float = 1.0,             # Used in Option B (e.g., 59.94/60)
    save_images: bool = True,
    include_last: bool = False,
    make_map_logs: bool = True,     # Generate mapping/duplicate reports
):
    folder = os.path.dirname(video_path)
    base   = os.path.splitext(os.path.basename(video_path))[0]
    outdir = os.path.join(folder, f"{base}_frames")
    if save_images:
        ensure_dir(outdir)

    # log files
    map_csv_path = os.path.join(folder, f"{base}_framemap.csv")
    dup_csv_path = os.path.join(folder, f"{base}_duplicates.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERR] open fail: {video_path}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if fps <= 0 or total <= 0:
        print(f"[ERR] bad meta fps={fps}, total={total}")
        cap.release()
        return

    # Build target seconds list
    if mode == "csv_len":
        if not csv_for_len:
            csv_for_len = find_label_csv_for_video(video_path)
        if not csv_for_len or not os.path.isfile(csv_for_len):
            print(f"[ERR] CSV length mode but CSV not found: {csv_for_len}")
            cap.release()
            return
        secs = secs_from_csv_len(csv_for_len, start_sec=start_sec, step_sec=step_sec)
        print(f"[A] Forced to CSV length: {len(secs)}초 (예: {secs[:5]} ...)")
    else:
        secs = secs_from_video(total, fps, start_sec=start_sec, step_sec=step_sec)
        print(f"[Default] Based on video length: {len(secs)}초 (예: {secs[:5]} ...)")
        if mode == "scaled_time":
            print(f"[B] Using scale correction: scale={scale:.6f}")

    if not secs:
        print(f"[WARN] no seconds to sample: {video_path}")
        cap.release()
        return

    # Frame duration (ms), tolerance for stability
    frame_ms = 1000.0 / max(fps, 1.0)
    eps_ms   = max(frame_ms / 3.0, 5.0)

   # Sequential streaming + nearest frame selection
    seq_idx  = 0
    next_i   = 0
    next_sec = secs[next_i]
    next_ms  = ideal_ms_of(next_sec, offset_ms, scale if mode == "scaled_time" else 1.0)

    prev_frame, prev_ms, prev_idx = None, None, None
    mappings = []  # 로깅용

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cur_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        cur_ms  = cap.get(cv2.CAP_PROP_POS_MSEC)
        if cur_ms < 0:
            cur_ms = (cur_idx / max(fps, 1.0)) * 1000.0

        while next_i < len(secs) and cur_ms >= (next_ms - eps_ms):
            # prev vs cur 중 목표 ms에 더 가까운 프레임 선택
            if prev_frame is None:
                chosen_frame = frame
                chosen_idx   = cur_idx
                chosen_ms    = cur_ms
                chosen_src   = "cur"
            else:
                d_prev = abs(prev_ms - next_ms)
                d_cur  = abs(cur_ms  - next_ms)
                if d_prev <= d_cur:
                    chosen_frame, chosen_idx, chosen_ms, chosen_src = prev_frame, prev_idx, prev_ms, "prev"
                else:
                    chosen_frame, chosen_idx, chosen_ms, chosen_src = frame, cur_idx, cur_ms, "cur"

            out_name = f"{base}_{seq_idx:06d}.png"  # START_SEC=1 → 1초=000000
            if save_images:
                cv2.imwrite(os.path.join(outdir, out_name), chosen_frame)

            mappings.append({
                "sec": next_sec,
                "saved_idx": seq_idx,
                "actual_frame_index": chosen_idx,
                "pos_msec": round(chosen_ms, 3),
                "chosen": chosen_src,
                "dist_ms": round(chosen_ms - next_ms, 3)
            })
            seq_idx += 1

            next_i += 1
            if next_i >= len(secs): break
            next_sec = secs[next_i]
            next_ms  = ideal_ms_of(next_sec, offset_ms, scale if mode == "scaled_time" else 1.0)

        prev_frame, prev_ms, prev_idx = frame, cur_ms, cur_idx
        if next_i >= len(secs): break

    # Padding missing seconds with last frame (important for Option A)
    while next_i < len(secs) and prev_frame is not None:
        out_name = f"{base}_{seq_idx:06d}.png"
        if save_images:
            cv2.imwrite(os.path.join(outdir, out_name), prev_frame)
        target_ms = ideal_ms_of(secs[next_i], offset_ms, scale if mode == "scaled_time" else 1.0)
        mappings.append({
            "sec": secs[next_i],
            "saved_idx": seq_idx,
            "actual_frame_index": prev_idx,
            "pos_msec": round(prev_ms, 3),
            "chosen": "tail",
            "dist_ms": round(prev_ms - target_ms, 3)
        })
        seq_idx += 1
        next_i  += 1

    # Add last frame (optional)
    if include_last and total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
        ret, last = cap.read()
        if ret:
            out_name = f"{base}_{seq_idx:06d}.png"
            if save_images:
                cv2.imwrite(os.path.join(outdir, out_name), last)
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if ms < 0:
                ms = ((total - 1) / max(fps, 1.0)) * 1000.0
            mappings.append({
                "sec": None,
                "saved_idx": seq_idx,
                "actual_frame_index": total - 1,
                "pos_msec": round(ms, 3),
                "chosen": "last",
                "dist_ms": None
            })
            seq_idx += 1

    cap.release()

    # Save logs / duplicate reports
    if make_map_logs:
        with open(map_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sec","saved_idx","actual_frame_index","pos_msec","chosen","dist_ms"])
            w.writeheader()
            for row in mappings:
                w.writerow(row)

        from collections import defaultdict
        dup = defaultdict(list)
        for m in mappings:
            dup[m["actual_frame_index"]].append(m["sec"])
        duplicates = [{"actual_frame_index": k,
                       "count": len(v),
                       "secs": ",".join(str(s) for s in v if s is not None)}
                      for k, v in dup.items() if len(v) > 1]
        with open(dup_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["actual_frame_index","count","secs"])
            w.writeheader()
            for r in sorted(duplicates, key=lambda x: (-x["count"], x["actual_frame_index"])):
                w.writerow(r)

    print(f"✅ frames saved: {seq_idx} → {outdir if save_images else '(images disabled)'}")
    if make_map_logs:
        print(f"   framemap: {map_csv_path}")
        print(f"   dup-report: {dup_csv_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="a")
    ap.add_argument("--start-sec", type=int, default=1)
    ap.add_argument("--step-sec", type=int, default=1)
    ap.add_argument("--mode", type=str, default="video", choices=["video","csv_len","scaled_time"],
                    help="video=video length, csv_len=force to CSV row count, scaled_time=scale correction")
    ap.add_argument("--offset-ms", type=float, default=0.0, help="timeline offset(ms)")
    ap.add_argument("--scale", type=float, default=1, help="scale correction (e.g., 59.94/60≈0.999)")
    ap.add_argument("--save-images", type=int, default=1)
    ap.add_argument("--include-last", type=int, default=0)
    args = ap.parse_args()

    for entry in sorted(os.listdir(args.root)):
        folder = os.path.join(args.root, entry)
        if not os.path.isdir(folder): continue
        videos = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                  if f.lower().endswith(VIDEO_EXTS)]
        for v in videos:
            csv_hint = find_label_csv_for_video(v)
            extract_one(
                v,
                start_sec=args.start_sec,
                step_sec=args.step_sec,
                mode=args.mode,
                csv_for_len=csv_hint,
                offset_ms=args.offset_ms,
                scale=args.scale,
                save_images=bool(args.save_images),
                include_last=bool(args.include_last),
                make_map_logs=True,
            )

if __name__ == "__main__":
    main()
