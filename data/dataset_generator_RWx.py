import os
import re
import json
import glob
import numpy as np
import scipy.io as sio
from scipy.io import loadmat

# ---------------------------
# Helpers for loading RWx MAT
# ---------------------------

def _to_str(x):
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return str(x)
    return str(x)


def load_rwx(path):
    """Load an RWx .mat file and extract step records + metadata.

    Expects a MATLAB struct array at m["data"].step with fields:
    - type, comment, relativeTime, current, voltage, temperature
    """
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    data = m["data"]
    steps = data.step  # array of mat_struct with fields
    # Normalize strings (bytes -> str) and strip
    types = [_to_str(s.type).strip() for s in steps]
    comments = [_to_str(s.comment).strip() for s in steps]
    return steps, types, comments


def find_rw_blocks(comments):
    mask = np.array([
        ("random walk" in c.lower()) or ("pulsed load" in c.lower())
        for c in comments
    ])
    blocks, start = [], None
    for i, is_rw in enumerate(mask):
        if is_rw and start is None:
            start = i
        elif (not is_rw) and start is not None:
            blocks.append((start, i - 1))
            start = None
    if start is not None:
        blocks.append((start, len(mask) - 1))
    return blocks


def first_n_pulses(block, types, n=100):
    s, e = block
    idxs = []
    for i in range(s, e + 1):
        t = types[i].strip().upper()
        if t in ("C", "D"):
            idxs.append(i)
            if len(idxs) >= n:
                break
    return idxs


def pulse_to_arrays(step):
    t = np.asarray(step.relativeTime, dtype=float)  # seconds from start of step
    i = np.asarray(step.current, dtype=float)       # Amps (D>0, C<0)
    v = np.asarray(step.voltage, dtype=float)       # Volts
    T = np.asarray(step.temperature, dtype=float)   # °C
    if t.ndim == 0:
        # Scalar step, synthesize a single sample at t=0
        t = np.array([0.0])
        i = np.array([float(i)])
        v = np.array([float(v)])
        T = np.array([float(T)])
    else:
        # Dedup near-identical timestamps
        mask = np.concatenate([[True], np.diff(t) > 1e-9])
        t, i, v, T = t[mask], i[mask], v[mask], T[mask]
    return t, i, v, T


def resample_concat(steps, idxs, Cn_Ah=2.1, Ts_s=1.0):
    """Resample each selected pulse on a Ts_s grid (ZOH for I, linear for V/T) and
    coulomb-count SoC (starts at 1.0 for this sequence). Concatenate results.
    """
    I_all, V_all, Temp_all, SoC_all = [], [], [], []
    soc = 1.0  # initial SoC
    for idx in idxs:
        t, i, v, T = pulse_to_arrays(steps[idx])
        t0, t1 = float(t[0]), float(t[-1])

        # Ts-aware grid aligned to multiples of Ts_s within [t0, t1]
        start = np.ceil(t0 / Ts_s) * Ts_s
        end = np.floor(t1 / Ts_s) * Ts_s
        if end < start:
            # extremely short pulse: sample at t0 only
            grid = np.array([t0])
        else:
            grid = np.arange(start, end + 1e-12, Ts_s)

        # ZOH for current
        prev_idx = np.searchsorted(t, grid, side="right") - 1
        prev_idx = np.clip(prev_idx, 0, len(i) - 1)
        I_zoh = i[prev_idx]

        # Linear interp for voltage and temperature
        V_lin = np.interp(grid, t, v)
        T_lin = np.interp(grid, t, T)

        # dt for SoC update (first step ~Ts_s, then exact diffs)
        dt = np.diff(np.r_[grid[0] - Ts_s, grid])
        dSoC = -I_zoh * dt / (Cn_Ah * 3600.0)
        soc_series = np.clip(soc + np.cumsum(dSoC), 0.0, 1.0)
        soc = float(soc_series[-1])

        I_all.append(I_zoh.reshape(-1, 1))
        V_all.append(V_lin.reshape(-1, 1))
        Temp_all.append(T_lin.reshape(-1, 1))
        SoC_all.append(soc_series.reshape(-1, 1))

    I_all = np.vstack(I_all)
    V_all = np.vstack(V_all)
    Temp_all = np.vstack(Temp_all)
    SoC_all = np.vstack(SoC_all)
    Time_idx = np.arange(I_all.shape[0]).reshape(-1, 1)
    return Time_idx, I_all, V_all, Temp_all, SoC_all


def build_train_test(rw_path, out_train, out_test, n_pulses=100, Cn=2.1, Ts=1.0):
    steps, types, comments = load_rwx(rw_path)
    blocks = find_rw_blocks(comments)
    if len(blocks) < 2:
        raise ValueError(
            f"Expected ≥2 random-walk blocks, found {len(blocks)}. Comments that matched: "
            f"{[c for c in comments if ('random walk' in c.lower()) or ('pulsed load' in c.lower())]}"
        )
    train_idxs = first_n_pulses(blocks[0], types, n=n_pulses)
    test_idxs = first_n_pulses(blocks[1], types, n=n_pulses)

    tr_Time, tr_I, tr_V, tr_T, tr_SoC = resample_concat(steps, train_idxs, Cn, Ts)
    ts_Time, ts_I, ts_V, ts_T, ts_SoC = resample_concat(steps, test_idxs, Cn, Ts)

    # Keep keys consistent with your original files (Temp key kept as-is)
    sio.savemat(out_train, {
        "tr_Time": tr_Time,
        "tr_Iin": tr_I,
        "tr_SoC": tr_SoC,
        "tr_Vout": tr_V,
        "Temp": tr_T,
        "Cn": float(Cn),
        "Ts": float(Ts),
    })
    sio.savemat(out_test, {
        "ts_Time": ts_Time,
        "ts_Iin": ts_I,
        "ts_SoC": ts_SoC,
        "ts_Vout": ts_V,
        "Temp": ts_T,
        "Cn": float(Cn),
        "Ts": float(Ts),
    })

    return {
        "rw_path": os.path.abspath(rw_path),
        "train_out": os.path.abspath(out_train),
        "test_out": os.path.abspath(out_test),
        "train_samples": int(tr_Time.shape[0]),
        "test_samples": int(ts_Time.shape[0]),
        "n_train_pulses": len(train_idxs),
        "n_test_pulses": len(test_idxs),
        "Ts": float(Ts),
        "Cn": float(Cn),
        "n_blocks": int(len(blocks)),
    }


# ---------------------------
# Interactive wrapper / CLI
# ---------------------------

def _default_output_dir():
    # NOTE: path contains spaces, keep exact spelling as requested
    return os.path.join("ENNC", "Ststem Identification", "Models")


def _extract_rw_tag(path):
    """Return something like 'RW9' from a filename, or fallback to 'RWx'."""
    m = re.search(r"(RW\d+)", os.path.basename(path), flags=re.IGNORECASE)
    return m.group(1).upper() if m else "RWx"


def interactive_pick_file():
    print("\n=== RWx Dataset Picker ===")
    candidates = sorted(set(glob.glob("RW*.mat") + glob.glob("*RW*.mat")))
    if candidates:
        for i, f in enumerate(candidates, 1):
            print(f"  [{i}] {f}")
        raw = input("Select a number, type a file path, or type a tag like RW9: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(candidates):
                return os.path.abspath(candidates[idx - 1])
        # If user typed a tag like RW9, try to match
        if re.fullmatch(r"RW\d+", raw, flags=re.IGNORECASE):
            for f in candidates:
                if raw.upper() in os.path.basename(f).upper():
                    return os.path.abspath(f)
        # Otherwise treat as a path
        if raw:
            return os.path.abspath(raw)
    else:
        # No candidates; ask for a path
        raw = input("Enter path to an RWx .mat file: ").strip()
        if raw:
            return os.path.abspath(raw)
    raise SystemExit("No dataset selected.")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Build train/test datasets from an RWx MATLAB file. Interactively asks which dataset if --rw is omitted.")
    ap.add_argument("--rw", help="Path to RWx .mat file (omit to get an interactive picker)")
    ap.add_argument("--pulses", type=int, default=100, help="Number of C/D pulses from each RW block")
    ap.add_argument("--Cn", type=float, default=2.1, help="Cell nominal capacity in Ah")
    ap.add_argument("--Ts", type=float, default=1.0, help="Sampling period in seconds")
    ap.add_argument("--outdir", default=_default_output_dir(), help="Output directory for train/test files")
    ap.add_argument("--tag", help="Override RW tag used in filenames (e.g., RW9)")
    ap.add_argument("--yes", action="store_true", help="Do not ask for confirmation before writing")
    args = ap.parse_args()

    rw_path = args.rw or interactive_pick_file()
    rw_tag = (args.tag or _extract_rw_tag(rw_path)).upper()

    outdir = os.path.normpath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    train_out = os.path.join(outdir, f"train{rw_tag}.mat")
    test_out = os.path.join(outdir, f"test{rw_tag}.mat")

    if not args.yes:
        print("\nSummary:")
        print(f"  RW file : {rw_path}")
        print(f"  Tag     : {rw_tag}")
        print(f"  Pulses  : {args.pulses}")
        print(f"  Cn/Ts   : {args.Cn} Ah / {args.Ts} s")
        print(f"  Outputs :\n    {train_out}\n    {test_out}")
        ok = input("Proceed? [Y/n] ").strip().lower()
        if ok and ok not in ("y", "yes"):  # default yes
            raise SystemExit("Aborted by user.")

    info = build_train_test(rw_path, train_out, test_out, args.pulses, args.Cn, args.Ts)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
