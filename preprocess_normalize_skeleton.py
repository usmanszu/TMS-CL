import os
import json
import argparse
import numpy as np
from tqdm import tqdm

# If you keep rotation helpers in preprocess/rotation.py:
from rotation import rotation_matrix, angle_between

def pre_normalization(
    data: np.ndarray,
    zaxis=(0, 1),            # bone to align with +Z (e.g., hip(0) -> spine(1) for NTU)
    xaxis=(8, 4),            # bone to align with +X (e.g., R-shoulder(8) -> L-shoulder(4) for NTU)
    fill_mode="repeat"       # how to pad null frames: ["repeat", "zero"]
):
    """
    data: (N, C=3, T, V, M)
    Returns normalized array of same shape.
    """
    assert data.ndim == 5, f"Expected (N,C,T,V,M), got {data.shape}"
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])   # (N,M,T,V,C)

    # 1) Fill null frames
    for i_s, skeleton in enumerate(tqdm(s, desc="Fill null frames")):
        if skeleton.sum() == 0:
            continue
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            # If first valid frame is not at t=0, shift up
            if person[0].sum() == 0:
                valid = (person.sum(-1).sum(-1) != 0)
                tmp = person[valid].copy()
                person[...] = 0
                person[:len(tmp)] = tmp
            # Backfill remaining all-zero tail
            for i_f in range(T):
                if person[i_f].sum() == 0:
                    if person[i_f:].sum() == 0:
                        if fill_mode == "repeat" and i_f > 0:
                            rest = T - i_f
                            num = int(np.ceil(rest / i_f))
                            pad = np.concatenate([person[:i_f] for _ in range(num)], 0)[:rest]
                            s[i_s, i_p, i_f:] = pad
                        # else keep zeros
                        break

    # 2) Subtract main body center (joint 1 for NTU)
    for i_s, skeleton in enumerate(tqdm(s, desc="Centering at joint #1")):
        if skeleton.sum() == 0:
            continue
        main_center = skeleton[0][:, 1:2, :].copy()    # (T,1,C)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_center) * mask

    # 3) Align z-axis: bone (zaxis[0] -> zaxis[1]) to +Z
    for i_s, skeleton in enumerate(tqdm(s, desc="Aligning Z-axis bone")):
        if skeleton.sum() == 0:
            continue
        j0 = skeleton[0, 0, zaxis[0]]
        j1 = skeleton[0, 0, zaxis[1]]
        v  = j1 - j0
        axis = np.cross(v, np.array([0., 0., 1.], dtype=np.float32))
        ang  = angle_between(v, np.array([0., 0., 1.], dtype=np.float32))
        Rz = rotation_matrix(axis, ang)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            nonzero = (person.sum(-1) != 0)
            for i_f in np.where(nonzero.any(axis=1))[0]:
                for i_j in np.where(nonzero[i_f])[0]:
                    s[i_s, i_p, i_f, i_j] = Rz @ person[i_f, i_j]

    # 4) Align x-axis: bone (xaxis[0] -> xaxis[1]) to +X
    for i_s, skeleton in enumerate(tqdm(s, desc="Aligning X-axis bone")):
        if skeleton.sum() == 0:
            continue
        jr = skeleton[0, 0, xaxis[0]]
        jl = skeleton[0, 0, xaxis[1]]
        v  = jr - jl
        axis = np.cross(v, np.array([1., 0., 0.], dtype=np.float32))
        ang  = angle_between(v, np.array([1., 0., 0.], dtype=np.float32))
        Rx = rotation_matrix(axis, ang)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            nonzero = (person.sum(-1) != 0)
            for i_f in np.where(nonzero.any(axis=1))[0]:
                for i_j in np.where(nonzero[i_f])[0]:
                    s[i_s, i_p, i_f, i_j] = Rx @ person[i_f, i_j]

    out = np.transpose(s, [0, 4, 2, 3, 1])   # (N,C,T,V,M)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-npy',  required=True, help='Input raw skeleton .npy (N,C,T,V,M)')
    ap.add_argument('--out-npy', required=True, help='Output normalized .npy')
    ap.add_argument('--zaxis', nargs=2, type=int, default=(0,1), help='bone i->j to align with +Z')
    ap.add_argument('--xaxis', nargs=2, type=int, default=(8,4), help='bone i->j to align with +X')
    ap.add_argument('--fill-mode', choices=['repeat','zero'], default='repeat')
    args = ap.parse_args()

    x = np.load(args.in_npy, mmap_mode='r')
    x = np.asarray(x, dtype=np.float32)
    y = pre_normalization(x, zaxis=tuple(args.zaxis), xaxis=tuple(args.xaxis), fill_mode=args.fill_mode)
    os.makedirs(os.path.dirname(args.out_npy), exist_ok=True)
    np.save(args.out_npy, y)
    print(f"[OK] saved: {args.out_npy}  shape={y.shape}")

if __name__ == '__main__':
    main()