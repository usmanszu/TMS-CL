# import os
# import glob
# from shutil import copyfile
#
# # ==============================
# # Configuration
# # ==============================
# all_path   = '../data/UAVHuman/ActionRecognition/Skeleton'   # source folder
# train_path = '../data/UAVHuman/ActionRecognition/Skeleton/train'
# test_path  = '../data/UAVHuman/ActionRecognition/Skeleton/test'
#
# # Make sure output dirs exist
# os.makedirs(train_path, exist_ok=True)
# os.makedirs(test_path, exist_ok=True)
#
# # Get all skeleton .txt files (case-insensitive)
# skeleton_filenames = sorted(
#     [os.path.basename(f) for f in glob.glob(os.path.join(all_path, '*.txt'))] +
#     [os.path.basename(f) for f in glob.glob(os.path.join(all_path, '*.TXT'))]
# )
#
# print(f"Found {len(skeleton_filenames)} skeleton files in {all_path}")
#
# # ==============================
# # Training IDs (V2 protocol)
# # ==============================
# train_list = [
#     0, 3, 4, 5, 6, 8, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 24, 26, 29, 30, 31, 32, 35,
#     36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 49, 52, 54, 56, 57, 59, 60, 61, 62, 63, 64, 66,
#     67, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 83, 84, 86, 87, 88, 89, 91, 92, 93,
#     94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113,
#     114, 115, 117, 118
# ]
#
# # ==============================
# # Split & Copy
# # ==============================
# train_count, test_count = 0, 0
#
# for basename in skeleton_filenames:
#     # Extract person ID from filename e.g. P001S01G01...
#     try:
#         pid = int(basename[1:4])
#     except:
#         print(f"[Warning] Could not extract person ID from: {basename}")
#         continue
#
#     src = os.path.join(all_path, basename)
#     if pid in train_list:
#         dst = os.path.join(train_path, basename)
#         train_count += 1
#     else:
#         dst = os.path.join(test_path, basename)
#         test_count += 1
#
#     copyfile(src, dst)
#
# # ==============================
# # Summary
# # ==============================
# print(f"\n✅ Split complete!")
# print(f"Train samples: {train_count}")
# print(f"Test samples : {test_count}")
# print(f"Train folder : {train_path}")
# print(f"Test folder  : {test_path}")

# next to create outside folders
# make_cross_subject_splits.py
import os
import glob
from shutil import copyfile

# ==============================
# Config
# ==============================
SRC_ALL_PATH = '../data/UAVHuman/ActionRecognition/Skeleton'   # source root (recursive)
OUT_V1_ROOT  = '../data/UAVHuman/benchmarks/uav-cross-subjectv1'
OUT_V2_ROOT  = '../data/UAVHuman/benchmarks/uav-cross-subjectv2'

# ==============================
# Train ID lists (person IDs)
# ==============================
TRAIN_LIST_V1 = {
    0, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27,
    28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 55, 56, 57, 59, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71,
    73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 98, 100, 102,
    103, 105, 106, 110, 111, 112, 114, 115, 116, 117, 118
}
TRAIN_LIST_V2 = {
    0, 3, 4, 5, 6, 8, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 24, 26, 29, 30, 31,
    32, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 49, 52, 54, 56, 57, 59, 60, 61,
    62, 63, 64, 66, 67, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 83, 84, 86,
    87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115, 117, 118
}

# ==============================
# Helpers
# ==============================
def collect_txts(root):
    pats = [os.path.join(root, '**', '*.txt'), os.path.join(root, '**', '*.TXT')]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(files))

def ensure_out(root):
    os.makedirs(os.path.join(root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(root, 'test'),  exist_ok=True)

def pid_from_name(basename):
    # expects names like P001S01G... -> pid=1
    try:
        return int(basename[1:4])
    except Exception:
        return None

def split_copy(files, out_root, train_ids, skip_if_exists=True):
    ensure_out(out_root)
    out_train = os.path.join(out_root, 'train')
    out_test  = os.path.join(out_root, 'test')

    n_train = n_test = n_skip = 0
    for src in files:
        base = os.path.basename(src)
        pid = pid_from_name(base)
        if pid is None:
            n_skip += 1
            continue
        dst = os.path.join(out_train if pid in train_ids else out_test, base)
        if skip_if_exists and os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        copyfile(src, dst)
        if pid in train_ids:
            n_train += 1
        else:
            n_test += 1
    return n_train, n_test, n_skip

# ==============================
# Main
# ==============================
if __name__ == '__main__':
    files = collect_txts(SRC_ALL_PATH)
    print(f"[INFO] Found {len(files)} .txt files under {SRC_ALL_PATH}")
    if not files:
        raise SystemExit("[ERROR] No skeleton .txt files found.")

    # Cross-Subject v1
    t1, s1, k1 = split_copy(files, OUT_V1_ROOT, TRAIN_LIST_V1, skip_if_exists=True)
    print(f"\n[OK] uav-cross-subjectv1 → {OUT_V1_ROOT}")
    print(f"  Train files: {t1}")
    print(f"  Test  files: {s1}")
    if k1: print(f"  Skipped (bad filename): {k1}")

    # Cross-Subject v2
    t2, s2, k2 = split_copy(files, OUT_V2_ROOT, TRAIN_LIST_V2, skip_if_exists=True)
    print(f"\n[OK] uav-cross-subjectv2 → {OUT_V2_ROOT}")
    print(f"  Train files: {t2}")
    print(f"  Test  files: {s2}")
    if k2: print(f"  Skipped (bad filename): {k2}")

    print("\n✅ Done.")
    print(f"uav-cross-subjectv1 layout:")
    print(f"  {OUT_V1_ROOT}/train (txt), {OUT_V1_ROOT}/test (txt)")
    print(f"uav-cross-subjectv2 layout:")
    print(f"  {OUT_V2_ROOT}/train (txt), {OUT_V2_ROOT}/test (txt)")

