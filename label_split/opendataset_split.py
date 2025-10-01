import argparse
import sys
from pathlib import Path
import shutil
import pandas as pd
from typing import Iterable, List, Dict, DefaultDict,Tuple
from collections import defaultdict
import glob
import re
# ---- Dataset / UUID guards ----
UUID_RE = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)
DATASET_KEYWORDS_DEFAULT = ['hyper-kvasir', 'kvasir', 'ldpolyps', 'nerthus']

def looks_like_uuid_name(name: str) -> bool:
    return bool(UUID_RE.search(name or ""))

def extract_dataset_hints(raw_path: str, excel_name: str, extra_keywords: List[str] | None = None) -> List[str]:
    """
    CSV path/excel 파일명에서 데이터셋 힌트를 추출한다.
    예: 'hyper-kvasir', 'ldpolyps', 'nerthus'
    """
    base = (raw_path or '') + ' ' + (excel_name or '')
    low = base.lower()
    kws = DATASET_KEYWORDS_DEFAULT + (extra_keywords or [])
    # 중복 제거 유지
    hints = []
    for kw in kws:
        if kw and kw not in hints and kw in low:
            hints.append(kw)
    return hints

def dataset_ok(path: Path, hints: List[str]) -> bool:
    """
    후보 경로가 CSV에서 추출한 데이터셋 힌트(hints)를 하나라도 포함하면 OK.
    힌트가 비어 있으면(판단 불가 시) 통과.
    """
    if not hints:
        return True
    plow = str(path).lower()
    return any(kw in plow for kw in hints)
# ========== 경로 보정 유틸 ==========

def squash_duplicated_segments(parts):
    out = []
    for seg in parts:
        if not out or out[-1].lower() != seg.lower():
            out.append(seg)
    return out

def map_output_segments(parts):
    """
    - 'output' → 제거
    - 'outputN' → 'N' (숫자만 남김)
    """
    mapped = []
    for seg in parts:
        low = seg.lower()
        if low == "output":
            continue  # 완전히 제거
        m = re.fullmatch(r"output(\d+)", low)
        if m:
            mapped.append(m.group(1))
        else:
            mapped.append(seg)
    return mapped

def canonicalize_path_like(p: Path) -> Path:
    """중복/불필요 세그먼트 정리 + output/outputN 변환. Windows anchor 안전 처리."""
    parts = list(p.parts)
    anchor = p.anchor  # 'D:\\' 또는 '/'
    if anchor:
        parts_wo_anchor = parts[1:] if parts and parts[0] == anchor else [seg for seg in parts if seg != anchor]
    else:
        parts_wo_anchor = parts
    parts_wo_anchor = squash_duplicated_segments(parts_wo_anchor)
    parts_wo_anchor = map_output_segments(parts_wo_anchor)
    return Path(anchor).joinpath(*parts_wo_anchor) if anchor else Path(*parts_wo_anchor)

def path_should_strip_output(excel_path: Path, row_path: str, mode: str, keywords: List[str]) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    base = excel_path.name.lower()
    rowp = (row_path or "").lower()
    for kw in keywords:
        kw = kw.strip().lower()
        if kw and (kw in base or kw in rowp):
            return True
    return False

# ========== 파일명 매칭 유틸 ==========

def normalize_case(s: str) -> str:
    return str(s).strip().lower()

def extract_core_tail(name: str) -> str:
    """
    데이터셋별 접두사가 길어도 실제 파일과 매칭되는 '핵심 꼬리'만 추출.
    우선순위:
    - Hyper-Kvasir: *_tool_<uuid>.ext  -> 'tool_<uuid>.ext'
    - LDPolyps: (informative|uninformative)_####.ext
    - 일반 숫자 꼬리: *_####.ext
    - Nerthus: 'bowel_'부터 or score 패턴
    - 마지막 3~4 토큰 또는 뒤 60자
    """
    s = name.strip()
    low = s.lower()

    # 0) Hyper-Kvasir: *_tool_<uuid>.jpg|png|jpeg
    m = re.search(r'(tool_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.(?:png|jpg|jpeg))$', low, re.IGNORECASE)
    if m:
        start = m.start(1)
        return s[start:]  # 'tool_<uuid>.ext'

    # 1) LDPolyps: informative_0153.jpg / uninformative_0153.png ...
    m = re.search(r'((?:informative|uninformative)_[0-9]{2,8}\.(?:png|jpg|jpeg))$', low, re.IGNORECASE)
    if m:
        start = m.start(1)
        return s[start:]

    # 2) 일반 숫자 꼬리: ..._0153.jpg / ..._000153.png
    m = re.search(r'(_[0-9]{2,8}\.(?:png|jpg|jpeg))$', low, re.IGNORECASE)
    if m:
        start = m.start(1) + 1  # 언더스코어 제외
        return s[start:]

    # 3) Nerthus 잔존 규칙: 'bowel_'부터
    i = low.find("bowel_")
    if i != -1:
        return s[i:]

    # 4) score 패턴 (예: ..._score_3-0_00000125.jpg)
    m = re.search(r'([A-Za-z0-9\-]+_\d+_score_[A-Za-z0-9\-]+_\d{4,8}\.(png|jpg|jpeg))$', s, re.IGNORECASE)
    if m:
        return m.group(1)

    # 5) 언더스코어 기준 꼬리 3~4 토큰
    parts = s.split("_")
    if len(parts) >= 4:
        return "_".join(parts[-4:])

    # 6) 최후수단: 뒤 60자
    return s[-60:]


# ========== 인덱스 구성 (속도 핵심) ==========

def build_file_index(root: Path, exts: List[str]) -> Tuple[Dict[str, Path], DefaultDict[str, List[Path]], DefaultDict[int, List[Path]], DefaultDict[str, List[Path]]]:
    """
    - name_idx: 파일명(lower) -> Path
    - tail_idx: core_tail(lower) -> [Path]
    - number_idx: 끝 숫자(int) -> [Path]
    - uuid_idx: 순수 uuid(lower) -> [Path]
    """
    exts = [e.lower() for e in exts]
    name_idx: Dict[str, Path] = {}
    tail_idx: DefaultDict[str, List[Path]] = defaultdict(list)
    number_idx: DefaultDict[int, List[Path]] = defaultdict(list)
    uuid_idx: DefaultDict[str, List[Path]] = defaultdict(list)

    def extract_number_tail(fname: str) -> int | None:
        m = re.search(r'(\d{1,8})\.(png|jpg|jpeg|bmp|tif|tiff)$', fname.lower())
        return int(m.group(1)) if m else None

    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        total += 1

        nm = normalize_case(p.name)
        name_idx[nm] = p

        core = normalize_case(extract_core_tail(p.name))
        tail_idx[core].append(p)

        num = extract_number_tail(p.name)
        if num is not None:
            number_idx[num].append(p)

        muuid = UUID_RE.search(p.name)
        if muuid:
            uuid_idx[muuid.group(0).lower()].append(p)

    print(f"[INFO] Indexed files: total={total}, unique_names={len(name_idx)}, with_number_tail={sum(len(v) for v in number_idx.values())}, with_uuid={sum(len(v) for v in uuid_idx.values())}")
    return name_idx, tail_idx, number_idx, uuid_idx

# ========== 입력 CSV 모음 찾기 ==========

def iter_excels(args) -> Iterable[Path]:
    yielded = set()
    if args.excel:
        for p in args.excel:
            p = Path(p)
            if p.exists():
                rp = p.resolve()
                if rp not in yielded:
                    yielded.add(rp); yield rp
    if args.excel_glob:
        for pat in args.excel_glob:
            for s in glob.glob(pat, recursive=True):
                p = Path(s)
                if p.exists():
                    rp = p.resolve()
                    if rp not in yielded:
                        yielded.add(rp); yield rp
    if args.excel_dir and args.pattern:
        excel_dir = Path(args.excel_dir)
        for pat in [s.strip() for s in args.pattern.split(",") if s.strip()]:
            for s in glob.glob(str(excel_dir / pat), recursive=True):
                p = Path(s)
                if p.exists():
                    rp = p.resolve()
                    if rp not in yielded:
                        yielded.add(rp); yield rp

# ========== CSV 읽기 ==========

def read_table(excel_path: Path) -> pd.DataFrame:
    suffix = excel_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(excel_path)
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(excel_path, sep=sep)
    raise ValueError(f"Unsupported file type: {suffix}")

def safe_makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ========== 매칭 로직 (인덱스 우선) ==========

def resolve_src_by_index(
    excel_path: Path,
    raw_path: str,
    target_name: str,
    root: Path,
    name_idx: Dict[str, Path],
    tail_idx: DefaultDict[str, List[Path]],
    number_idx: DefaultDict[int, List[Path]],
    uuid_idx: DefaultDict[str, List[Path]],
    allow_substring: bool,
) -> Path | None:
    hints = extract_dataset_hints(raw_path, excel_path.name)

    nm = normalize_case(target_name)

    # 0) UUID 매칭 (Hyper-Kvasir 최우선) — rglob 없이 즉시 결정
    muuid = UUID_RE.search(target_name)
    if muuid:
        key = muuid.group(0).lower()
        candu = uuid_idx.get(key, [])
        candu = [p for p in candu if dataset_ok(p, hints)]
        if candu:
            if raw_path:
                raw_low = normalize_case(raw_path)
                candu = sorted(candu, key=lambda p: (0 if raw_low in normalize_case(str(p)) else 1, len(str(p))))
            return candu[0]

    # 1) 파일명 정확 일치
    hit = name_idx.get(nm)
    if hit and hit.exists() and dataset_ok(hit, hints):
        return hit

    # 2) 부분 일치 (core-tail)
    if allow_substring:
        core = normalize_case(extract_core_tail(target_name))
        cand = tail_idx.get(core, [])
        cand = [p for p in cand if dataset_ok(p, hints)]
        if cand:
            for p in cand:
                if normalize_case(p.name) == nm:
                    return p
            if raw_path:
                raw_low = normalize_case(raw_path)
                cand = sorted(cand, key=lambda p: (0 if raw_low in normalize_case(str(p)) else 1, len(str(p))))
            return cand[0]

    # 3) 숫자 꼬리 매칭 — 단, UUID가 있는 파일명은 숫자매칭 금지
    if not looks_like_uuid_name(target_name):
        mnum = re.search(r'(\d{1,8})\.(png|jpg|jpeg|bmp|tif|tiff)$', nm)
        if mnum:
            num = int(mnum.group(1))
            cand2 = number_idx.get(num, [])
            cand2 = [p for p in cand2 if dataset_ok(p, hints)]
            if cand2:
                if raw_path:
                    raw_low = normalize_case(raw_path)
                    cand2 = sorted(cand2, key=lambda p: (0 if raw_low in normalize_case(str(p)) else 1, len(str(p))))
                return cand2[0]

    # 4) 최후 수단 rglob — UUID가 있으면 이미 위에서 끝났으므로 여기선 UUID 없는 케이스만
    candidate = canonicalize_path_like(root / (raw_path or ""))
    search_root = candidate
    if not search_root.exists():
        while not search_root.exists() and search_root.parent != search_root:
            search_root = search_root.parent
        if not search_root.exists():
            search_root = root
    try:
        core_tail = normalize_case(extract_core_tail(target_name))
        best = None
        best_key = (1, 1e9)
        for q in search_root.rglob("*"):
            if not q.is_file():
                continue
            qn = normalize_case(q.name)
            if qn == nm and dataset_ok(q, hints):
                return q
            if allow_substring:
                if (core_tail in qn or qn.endswith(core_tail)) and dataset_ok(q, hints):
                    k = (0 if normalize_case(raw_path or "") in normalize_case(str(q)) else 1, len(str(q)))
                    if k < best_key:
                        best_key = k; best = q
        return best
    except (PermissionError, OSError):
        return None
# ========== 메인 처리 ==========

def process_excel(
    excel_path: Path,
    root: Path,
    dest: Path,
    path_col: str,
    name_col: str,
    label_col: str,
    split_col: str | None,
    split_fixed: str | None,
    strip_mode: str,
    strip_keywords: List[str],
    allow_substring: bool,
    ext_priority: List[str],  # (인덱스 기반이라 사실 필요성↓, 인터페이스 유지)
    dry_run: bool,
    name_idx: Dict[str, Path],
    tail_idx: DefaultDict[str, List[Path]],
    number_idx: DefaultDict[int, List[Path]],
    uuid_idx: DefaultDict[str, List[Path]],
):
    try:
        df = read_table(excel_path)
    except Exception as e:
        print(f"[ERROR] Read failed: {excel_path} -> {e}", file=sys.stderr)
        return (0, 0, 1)

    needed_cols = [path_col, name_col, label_col]
    for col in needed_cols:
        if col not in df.columns:
            print(f"[ERROR] {excel_path.name}: Column '{col}' not found. Columns={list(df.columns)}", file=sys.stderr)
            return (0, 0, 1)

    moved = skipped = errors = 0

    for idx, row in df.iterrows():
        raw_path = str(row[path_col]).strip()
        target_name = str(row[name_col]).strip()
        label = str(row[label_col]).strip()

        # Split 결정: split_col > split_fixed > 'train'
        split = None
        if split_col and split_col in df.columns:
            split = str(row[split_col]).strip() or None
        if not split and split_fixed:
            split = split_fixed.strip()
        if not split:
            split = "train"

        if not raw_path or not target_name:
            print(f"[WARN] {excel_path.name} Row {idx}: empty path or filename -> skip")
            skipped += 1
            continue

        # auto-strip 여부(경로 힌트 정규화에만 사용)
        if path_should_strip_output(excel_path, raw_path, strip_mode, strip_keywords):
            raw_path_use = str(canonicalize_path_like(Path(raw_path)))
        else:
            raw_path_use = raw_path

        # 인덱스 기반으로 빠르게 찾기
        src_file = resolve_src_by_index(
            excel_path=excel_path,
            raw_path=raw_path_use,
            target_name=target_name,
            root=root,
            name_idx=name_idx,
            tail_idx=tail_idx,
            number_idx=number_idx,
            uuid_idx=uuid_idx,     # ← 추가!
            allow_substring=allow_substring,
        )


        if src_file is None or not src_file.exists():
            print(f"[WARN] {excel_path.name} Row {idx}: source not found "
                  f"for '{target_name}' (path='{raw_path}')")
            skipped += 1
            continue

        try:
            dest_dir = dest / split / label   # <-- 상위에 split 폴더, 그 안에 label 폴더
            safe_makedirs(dest_dir)
            dest_path = dest_dir / target_name

            if dest_path.exists():
                try:
                    if src_file.resolve() == dest_path.resolve():
                        print(f"[INFO] {excel_path.name} Row {idx}: already at destination -> {dest_path}")
                        skipped += 1
                        continue
                except Exception:
                    pass
                stem, ext = dest_path.stem, dest_path.suffix
                k = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{stem}__dup{k}{ext}"
                    k += 1

            print(f"[MOVE] [{excel_path.name}#{idx}] {src_file} -> {dest_path}")
            if not dry_run:
                shutil.move(str(src_file), str(dest_path))
            moved += 1

        except Exception as e:
            print(f"[ERROR] {excel_path.name} Row {idx}: move failed -> {e}", file=sys.stderr)
            errors += 1

    print(f"[SUMMARY] {excel_path.name}: moved={moved}, skipped={skipped}, errors={errors}")
    return (moved, skipped, errors)

def main():
    ap = argparse.ArgumentParser(description="Ultra-fast batch rename & move using a prebuilt file index.")
    # 입력 소스
    ap.add_argument("--excel", action="append", help="Single Excel/CSV path. Can be provided multiple times.")
    ap.add_argument("--excel-glob", nargs="*", help="Glob pattern(s) for Excel/CSV.")
    ap.add_argument("--excel-dir", help="Directory containing Excel/CSV files.")
    ap.add_argument("--pattern", help="Comma-separated patterns inside --excel-dir.")

    # 공통 파라미터
    ap.add_argument("--root", required=True, type=Path, help="Root directory that actually contains image files.")
    ap.add_argument("--dest", required=True, type=Path, help="Destination root (files moved into '<dest>/<split>/<label>/').")
    ap.add_argument("--path-col", default="Path", help="Column containing source path hint (dir or file).")
    ap.add_argument("--name-col", default="Filename", help="Column containing desired filename.")
    ap.add_argument("--label-col", default="Label", help="Column containing label (used as subfolder).")

    # split 제어
    ap.add_argument("--split-col", default=None, help="Column name that contains split name (train/val/test/label).")
    ap.add_argument("--split-fixed", default=None, help="Fixed split name if there's no split column.")

    # strip-output 제어
    ap.add_argument("--strip-output-mode", choices=["never", "always", "auto"], default="auto",
                    help="When to strip 'output' segments in *hint* paths. auto: only if keyword matches.")
    ap.add_argument("--strip-keywords", default="nerthus",
                    help="Comma-separated keywords to trigger strip in auto mode.")

    # 매칭/기타
    ap.add_argument("--allow-substring", action="store_true", help="Allow substring match when exact filename not found.")
    ap.add_argument("--ext-priority", nargs="*", default=[".png", ".jpg", ".jpeg"], help="(kept for compatibility)")
    ap.add_argument("--dry-run", action="store_true", help="Log actions without moving files.")
    args = ap.parse_args()

    strip_keywords = [s.strip() for s in (args.strip_keywords or "").split(",") if s.strip()]

    # 0) 사전 인덱스 구축 (한 번만!)
    print(f"[INFO] Building file index under: {args.root}")
    name_idx, tail_idx, number_idx, uuid_idx = build_file_index(args.root, args.ext_priority)
    print(f"[INFO] Indexed files: {len(name_idx)} (unique names), {sum(len(v) for v in tail_idx.values())} (by core-tail)")


    total_moved = total_skipped = total_errors = 0
    excel_list = list(iter_excels(args))
    if not excel_list:
        print("[ERROR] No Excel/CSV files found from given inputs.", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Target Excel/CSV files ({len(excel_list)}):")
    for p in excel_list:
        print(f"  - {p}")

    for ex in excel_list:
        m, s, e = process_excel(
            excel_path=ex,
            root=args.root,
            dest=args.dest,
            path_col=args.path_col,
            name_col=args.name_col,
            label_col=args.label_col,
            split_col=args.split_col,
            split_fixed=args.split_fixed,
            strip_mode=args.strip_output_mode,
            strip_keywords=strip_keywords,
            allow_substring=args.allow_substring,
            ext_priority=args.ext_priority,
            dry_run=args.dry_run,
            name_idx=name_idx,
            tail_idx=tail_idx,
            number_idx=number_idx,
            uuid_idx=uuid_idx
        )
        total_moved += m
        total_skipped += s
        total_errors += e

    print(f"\n[TOTAL] moved={total_moved}, skipped={total_skipped}, errors={total_errors}")
    sys.exit(0 if total_errors == 0 else 1)

if __name__ == "__main__":
    main()

