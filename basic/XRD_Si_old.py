import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. 実験データの読み込み
# filename = 'C:\\Users\\kurot\\LaTeX\\20254Q\\gtICDDデータ類-20260105\\7-1_本焼.txt'
filename = "basic/data/gtICDDデータ類-20260106/7-1_本焼.txt"

data = []
with open(filename, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
df = pd.DataFrame(data, columns=["2theta", "Intensity"])

# 2. Si標準データの読み込みと補正
si_std_df = pd.read_csv(
    # "C:\\Users\\kurot\\LaTeX\\20254Q\\gtICDDデータ類-20260105\\Si_標準.csv",
    "basic/data/gtICDDデータ類-20260106/Si_標準.csv",
    encoding="cp932",
)
print("Si Standard Data Loaded")

correction_points = []
# Si標準データの各行について処理
for _, row in si_std_df.iterrows():
    # NaNチェック
    if pd.isna(row["2θ (度)"]):
        continue

    theo_2theta = row["2θ (度)"]
    h, k, l = row["h"], row["k"], row["l"]

    # 実験データから対応するピークを探す (±0.5度の範囲)
    window = df[
        (df["2theta"] >= theo_2theta - 0.5) & (df["2theta"] <= theo_2theta + 0.5)
    ]
    if not window.empty:
        # ウィンドウ内でピーク検出
        window_peaks_idx, _ = find_peaks(window["Intensity"], height=100)
        if len(window_peaks_idx) > 0:
            # 最も強いピークを採用
            best_idx = window.iloc[window_peaks_idx]["Intensity"].idxmax()
            obs_2theta = df.loc[best_idx, "2theta"]
            correction_points.append(
                {
                    "theo": theo_2theta,
                    "obs": obs_2theta,
                    "hkl": f"Si ({int(h)}{int(k)}{int(l)})",
                }
            )

print(f"Correction points used: {len(correction_points)}")

# 補正関数の作成
if len(correction_points) > 1:
    obs = np.array([p["obs"] for p in correction_points])
    theo = np.array([p["theo"] for p in correction_points])
    # 差分(theo - obs)を1次関数でフィッティング
    diff = theo - obs
    coeffs = np.polyfit(obs, diff, 1)
    correction_func = np.poly1d(coeffs)

    # --- 補正曲線のプロット (表示したい場合) ---
    # plt.figure(figsize=(8, 6))
    # plt.scatter(obs, diff, color='blue', label='Experimental Data')
    # x_range = np.linspace(obs.min() - 5, obs.max() + 5, 100)
    # plt.plot(x_range, correction_func(x_range), 'r--', label='Linear Fit')
    # plt.show()
    # ---------------------------------------

    # データを補正
    df["2theta_corr"] = df["2theta"] + correction_func(df["2theta"])
else:
    print("Warning: Not enough Si peaks found. No correction applied.")
    df["2theta_corr"] = df["2theta"]

# 3. 格子定数の算出 (BaTiO3)
# 44-46度の(002)/(200)分裂ピーク
roi = df[(df["2theta_corr"] >= 44.0) & (df["2theta_corr"] <= 46.0)]
roi_peaks_idx, _ = find_peaks(roi["Intensity"], height=800, distance=5)
roi_peaks = roi.iloc[roi_peaks_idx]

# 波長 (Cu Ka1)
lam = 1.54056
c_calc = 4.038  # Default
a_calc = 3.994  # Default

if len(roi_peaks) >= 2:
    # 強度上位2つを取得
    top2 = roi_peaks.sort_values(by="Intensity", ascending=False).head(2)
    # 角度順にソート (低角=(002), 高角=(200))
    top2_sorted = top2.sort_values(by="2theta_corr")

    p002 = top2_sorted.iloc[0]["2theta_corr"]
    p200 = top2_sorted.iloc[1]["2theta_corr"]

    d002 = lam / (2 * np.sin(np.radians(p002 / 2)))
    d200 = lam / (2 * np.sin(np.radians(p200 / 2)))

    c_calc = 2 * d002
    a_calc = 2 * d200
    print(f"Calculated: a={a_calc:.4f}, c={c_calc:.4f}")

# 4. 指数付け (BaTiO3 Reference CSV使用)
ref_df = pd.read_csv("basic/data/gtICDDデータ類-20260106/1_BaTiO3-tetragonal.csv")
labels = []
# 強度が低いピークは除外
ref_strong = ref_df[ref_df["I(f)"] >= 5]  # 強度閾値5以上のピークのみ使用

for _, row in ref_strong.iterrows():
    h, k, l = int(row["h"]), int(row["k"]), int(row["l"])

    # 算出したa, cを使って理論位置を計算
    d_inv2 = (h**2 + k**2) / (a_calc**2) + (l**2) / (c_calc**2)
    if d_inv2 <= 0:
        continue
    d = np.sqrt(1 / d_inv2)

    if lam / (2 * d) > 1:
        continue
    theo_th = 2 * np.degrees(np.arcsin(lam / (2 * d)))

    # 近くの実験ピークを探す
    window = df[
        (df["2theta_corr"] >= theo_th - 0.5) & (df["2theta_corr"] <= theo_th + 0.5)
    ]
    if not window.empty:
        window_peaks_idx, _ = find_peaks(window["Intensity"], height=300)
        if len(window_peaks_idx) > 0:
            best_idx = window.iloc[window_peaks_idx]["Intensity"].idxmax()
            real_th = df.loc[best_idx, "2theta_corr"]
            inten = df.loc[best_idx, "Intensity"]

            labels.append(
                {
                    "label": f"({h}{k}{l})",
                    "x": real_th,
                    "y": inten,
                    "color": "blue",  # ★変更点: BaTiO3の色を指定
                }
            )

# ラベルの重なり処理
labels.sort(key=lambda k: k["x"])
unique_labels = []
if labels:
    curr = labels[0]
    for next_l in labels[1:]:
        if abs(next_l["x"] - curr["x"]) < 0.3:
            if next_l["label"] not in curr["label"]:
                curr["label"] += f"\n{next_l['label']}"
            if next_l["y"] > curr["y"]:
                curr["x"] = next_l["x"]
                curr["y"] = next_l["y"]
                # 色はcurr(最初に検出されたもの)を引き継ぎます
        else:
            unique_labels.append(curr)
            curr = next_l
    unique_labels.append(curr)

# Siラベルを追加
for p in correction_points:
    search_th = p["theo"]
    window = df[
        (df["2theta_corr"] >= search_th - 0.3) & (df["2theta_corr"] <= search_th + 0.3)
    ]
    if not window.empty:
        y_pos = window["Intensity"].max()
        unique_labels.append(
            {
                "label": p["hkl"],
                "x": search_th,
                "y": y_pos,
                "color": "red",  # ★変更点: Siの色を指定
            }
        )

# 5. プロット
plt.figure(figsize=(12, 8))
plt.plot(df["2theta_corr"], df["Intensity"], "k-", linewidth=0.8, label="Corrected XRD")

# ラベルの描画
for l in unique_labels:
    plt.text(
        l["x"],
        l["y"] + 500,
        l["label"],
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=8,
        color=l["color"],
    )  # ★変更点: 指定した色でテキストを描画

# ★変更点: 凡例用のダミープロットを追加
# 実際のデータはプロットせず、ラベルと色だけを凡例に登録します
plt.plot([], [], "o", color="blue", label="BaTiO3 Peaks", linestyle="None")
plt.plot([], [], "o", color="red", label="Si Peaks", linestyle="None")

plt.legend()
plt.xlabel(r"2$\theta$ (degrees)")
plt.ylabel("Intensity (a.u.)")
plt.xlim(20, 90)
plt.ylim(0, df["Intensity"].max() * 1.3)
plt.tight_layout()
plt.savefig("basic/out/7-1_本焼.png")
plt.show()
plt.close()
