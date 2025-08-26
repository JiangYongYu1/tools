#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# FP8 E4M3 (OCP OFP8) 解码函数
# -----------------------------
# 格式：1 sign, 4 exponent (bias=7), 3 mantissa
# 规则：
# - e=0, m=0 -> +0 / -0
# - e=0, m>0 -> 次正规：(-1)^s * 2^(1-bias) * (m/2^3)
# - 其他 e(含 e=15 且 m<=6) -> 正规：(-1)^s * 2^(e-bias) * (1 + m/2^3)
# - e=15, m=7 -> NaN
def decode_e4m3(byte: int):
    s = (byte >> 7) & 0x1
    e = (byte >> 3) & 0xF
    m = byte & 0x7
    bias = 7

    # NaN
    if e == 0xF and m == 0x7:
        return np.nan, {"sign": s, "exp": e, "mant": m, "class": "NaN"}

    # Zero
    if e == 0 and m == 0:
        val = -0.0 if s else 0.0
        return val, {"sign": s, "exp": e, "mant": m, "class": "zero"}

    # Subnormal
    if e == 0 and m > 0:
        frac = m / (2**3)
        val = ((-1) ** s) * (2 ** (1 - bias)) * frac  # 2^(1-bias) * (m/2^3)
        return val, {"sign": s, "exp": e, "mant": m, "class": "subnormal"}

    # Normal (包含 e==15 且 m<=6 的有限值；本规范无 ±Inf)
    frac = 1.0 + m / (2**3)
    val = ((-1) ** s) * (2 ** (e - bias)) * frac
    return val, {"sign": s, "exp": e, "mant": m, "class": "normal"}


def build_table():
    rows = []
    for b in range(256):
        v, meta = decode_e4m3(b)
        rows.append({
            "code": b,
            "sign": meta["sign"],
            "exp": meta["exp"],
            "mant": meta["mant"],
            "class": meta["class"],
            "value": v,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    finite = df[df["class"] != "NaN"]
    abs_lt_1 = finite[np.abs(finite["value"]) < 1]
    abs_le_1 = finite[np.abs(finite["value"]) <= 1]
    pos_lt_1 = finite[(finite["value"] > 0) & (finite["value"] < 1)]
    neg_gt_m1 = finite[(finite["value"] < 0) & (finite["value"] > -1)]

    # int8 在 [-1,1] 范围内的可表示值
    int8_vals = list(range(-128, 128))
    int8_le_1 = [x for x in int8_vals if abs(x) <= 1]  # {-1, 0, 1}
    int8_lt_1 = [x for x in int8_vals if abs(x) < 1]   # {0}

    summary = {
        "total_codes": int(len(df)),
        "nan_codes": int(df["class"].eq("NaN").sum()),
        "signed_zero_codes": int(df["class"].eq("zero").sum()),  # +0/-0
        "finite_codes": int(len(finite)),
        "|x|<1_count": int(len(abs_lt_1)),
        "|x|<=1_count": int(len(abs_le_1)),
        "positive_(0,1)_count": int(len(pos_lt_1)),
        "negative_(-1,0)_count": int(len(neg_gt_m1)),
        "int8_|x|<=1_count": len(int8_le_1),
        "int8_|x|<1_count": len(int8_lt_1),
        "fp8_vs_int8_extra_bins_(<=1)": int(len(abs_le_1)) - len(int8_le_1),
        "fp8_vs_int8_extra_bins_(<1)": int(len(abs_lt_1)) - len(int8_lt_1),
        "min_normal": 2 ** -6,
        "min_subnormal": 2 ** -9,
        "max_finite_abs": float(
            finite["value"].abs().max(skipna=True)
        ),
    }
    return summary


def plots(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    finite = df[df["class"] != "NaN"]

    # 全范围散点
    xs = finite["value"].to_numpy()
    ys = np.zeros_like(xs)
    plt.figure(figsize=(10, 2.2))
    plt.scatter(xs, ys, s=6)
    plt.title("FP8 E4M3: All finite representable values")
    plt.xlabel("value")
    plt.yticks([])
    plt.grid(True, which="both", axis="x")
    plt.xlim(-480, 480)
    plt.tight_layout()
    plt.savefig(outdir / "fp8_e4m3_all_values.png", dpi=160)
    plt.close()

    # 近零放大
    mask = (finite["value"] >= -1.2) & (finite["value"] <= 1.2)
    xs2 = finite.loc[mask, "value"].to_numpy()
    ys2 = np.zeros_like(xs2)
    plt.figure(figsize=(10, 2.2))
    plt.scatter(xs2, ys2, s=12)
    plt.title("FP8 E4M3: Zoomed near zero ([-1.2, 1.2])")
    plt.xlabel("value")
    plt.yticks([])
    plt.grid(True, which="both", axis="x")
    plt.tight_layout()
    plt.savefig(outdir / "fp8_e4m3_zoom.png", dpi=160)
    plt.close()

    # ULP 曲线（正数侧）
    pos_vals = np.sort(finite[finite["value"] > 0]["value"].to_numpy())
    ulp = np.empty_like(pos_vals)
    ulp[:-1] = pos_vals[1:] - pos_vals[:-1]
    ulp[-1] = np.nan
    plt.figure(figsize=(8, 4))
    plt.plot(pos_vals[:-1], ulp[:-1])
    plt.title("FP8 E4M3: ULP (forward spacing) vs value (positive)")
    plt.xlabel("value")
    plt.ylabel("ULP spacing")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "fp8_e4m3_ulp_curve.png", dpi=160)
    plt.close()


def export_csv_and_tex(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # 全量
    df.to_csv(outdir / "fp8_e4m3_all_values.csv", index=False)
    # 正数 ≤2.0 的便捷子表
    finite = df[df["class"] != "NaN"]
    pos_u2 = finite[(finite["value"] > 0) & (finite["value"] <= 2.0)].sort_values("value")
    pos_u2.to_csv(outdir / "fp8_e4m3_positive_values_upto_2.csv", index=False)

    # LaTeX 概要
    tex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l r}",
        r"\hline",
        rf"Total codes & 256 \\",
        rf"NaN codes & {int(df['class'].eq('NaN').sum())} \\",
        rf"Signed zeros & {int(df['class'].eq('zero').sum())} \;(\{{+0,-0\}}) \\",
        rf"Finite codes & {int((df['class']!='NaN').sum())} \\",
        rf"Range (finite) & [$-448, +448$] \\",
        rf"Min normal & $2^{{-6}} \approx {2**-6:.6f}$ \\",
        rf"Min subnormal & $2^{{-9}} \approx {2**-9:.6f}$ \\",
        r"\hline",
        r"\end{tabular}",
        r"\caption{FP8 E4M3 (OFP8) summary. No $\pm\infty$; two NaN encodings.}",
        r"\end{table}",
    ]
    (outdir / "fp8_e4m3_summary.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def main():
    outdir = Path("./fp8_e4m3_out")
    df = build_table()
    summary = summarize(df)

    print("==== FP8 E4M3 Summary ====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nNotes:")
    print("- 同范围 [-1,1] 对比：int8 仅有 {-1,0,1} 3 个值；E4M3 有 114 个编码（含 ±1），或 112 个编码（不含 ±1）。")
    print("- E4M3 最大有限值约 ±448；最小正规 2^-6；最小次正规 2^-9；无 ±Inf，只有两个 NaN。")

    plots(df, outdir)
    export_csv_and_tex(df, outdir)
    print(f"\n图表与 CSV/LaTeX 已输出到：{outdir.resolve()}")


if __name__ == "__main__":
    main()