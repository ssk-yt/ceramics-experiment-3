import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# 0. ラベルの重なり回避を関数化
def get_unique_labels(label_list, threshold=0.6):
    if not label_list:
        return []

    # x座標でソート
    label_list.sort(key=lambda k: k["x"])

    unique_labels = []
    curr = label_list[0].copy()

    for next_l in label_list[1:]:
        # しきい値より近い場合は統合
        if abs(next_l["x"] - curr["x"]) < threshold:
            if next_l["label"] not in curr["label"]:
                curr["label"] += f"\n{next_l['label']}"
            # 高い方のY座標を採用する
            if next_l["y"] > curr["y"]:
                curr["x"] = next_l["x"]
                curr["y"] = next_l["y"]
        else:
            unique_labels.append(curr)
            curr = next_l.copy()

    unique_labels.append(curr)
    return unique_labels


# 1. 実験データの読み込み
# filename = 'C:\\Users\\kurot\\LaTeX\\20254Q\\gtICDDデータ類-20260105\\7-1_本焼.txt'
sample_name = "7-1_仮焼"
filename = f"basic/data/gtICDDデータ類-20260106/{sample_name}.txt"
data = []
# ファイルが存在しない場合のエラーハンドリングを追加する場合、try-exceptブロックを使用しますが、
# 元のコードに従いそのまま記述します。
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
# si_std_df = pd.read_csv('C:\\Users\\kurot\\LaTeX\\20254Q\\gtICDDデータ類-20260105\\Si_標準.csv', encoding='cp932')
si_std_df = pd.read_csv(
    "basic/data/gtICDDデータ類-20260106/Si_標準.csv", encoding="cp932"
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

    # データを補正
    df["2theta_corr"] = df["2theta"] + correction_func(df["2theta"])
else:
    print("Warning: Not enough Si peaks found. No correction applied.")
    df["2theta_corr"] = df["2theta"]

# 3. 格子定数の算出 (BaTiO3) と誤差評価
# 44-46度の(002)/(200)分裂ピーク
roi = df[(df["2theta_corr"] >= 44.0) & (df["2theta_corr"] <= 46.0)]
roi_peaks_idx, _ = find_peaks(roi["Intensity"], height=800, distance=35)
roi_peaks = roi.iloc[roi_peaks_idx]

# 波長 (Cu Ka1)
lam = 1.54056
c_calc = 4.038  # Default
a_calc = 3.994  # Default

# 誤差計算用の設定
delta_2theta_deg = 0.02  # 読み取り精度 (2θの誤差)
delta_theta_rad = np.radians(delta_2theta_deg / 2)  # θの誤差 (ラジアン)

if len(roi_peaks) >= 2:
    # 強度上位2つを取得
    top2 = roi_peaks.sort_values(by="Intensity", ascending=False).head(2)
    # 角度順にソート (低角=(002), 高角=(200))
    top2_sorted = top2.sort_values(by="2theta_corr")

    p002 = top2_sorted.iloc[0]["2theta_corr"]
    p200 = top2_sorted.iloc[1]["2theta_corr"]

    # --- 格子定数の計算 ---
    theta002_rad = np.radians(p002 / 2)
    d002 = lam / (2 * np.sin(theta002_rad))
    c_calc = 2 * d002

    theta200_rad = np.radians(p200 / 2)
    d200 = lam / (2 * np.sin(theta200_rad))
    a_calc = 2 * d200

    # --- 誤差の計算 (誤差伝播則: Δd = d * cot(θ) * Δθ) ---
    # c軸の誤差
    delta_d002 = d002 * (1 / np.tan(theta002_rad)) * delta_theta_rad
    delta_c = 2 * delta_d002

    # a軸の誤差
    delta_d200 = d200 * (1 / np.tan(theta200_rad)) * delta_theta_rad
    delta_a = 2 * delta_d200

    print("-" * 50)
    print(f"Peak (002) 2theta: {p002:.4f} deg")
    print(f"Peak (200) 2theta: {p200:.4f} deg")
    print("-" * 50)
    print(
        f"Calculated Lattice Constants (with error from 2theta accuracy {delta_2theta_deg} deg):"
    )
    print(f"a = {a_calc:.5f} +/- {delta_a:.5f} Å")
    print(f"c = {c_calc:.5f} +/- {delta_c:.5f} Å")
    print("-" * 50)

# 4. 指数付け (BaTiO3 Reference CSV使用)
# ref_df = pd.read_csv('C:\\Users\\kurot\\LaTeX\\20254Q\\gtICDDデータ類-20260105\\1_BaTiO3-tetragonal.csv')
ref_df = pd.read_csv("basic/data/gtICDDデータ類-20260106/1_BaTiO3-tetragonal.csv")
# ref_df = pd.read_csv("basic/data/gtICDDデータ類-20260106/2_BaTiO3-hexagonal.csv")

# BaTiO3ラベルを追加
bt_labels = []
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
        window_peaks_idx, _ = find_peaks(window["Intensity"], height=300, distance=40)
        if len(window_peaks_idx) > 0:
            best_idx = window.iloc[window_peaks_idx]["Intensity"].idxmax()
            real_th = df.loc[best_idx, "2theta_corr"]
            inten = df.loc[best_idx, "Intensity"]

            bt_labels.append(
                {"label": f"({h}{k}{l})", "x": real_th, "y": inten, "color": "blue"}
            )

# Siラベルを追加
si_labels = []
for p in correction_points:
    search_th = p["theo"]
    if search_th <= 90:  # 90°以下のラベルのみ
        window = df[
            (df["2theta_corr"] >= search_th - 0.3)
            & (df["2theta_corr"] <= search_th + 0.3)
        ]
        if not window.empty:
            y_pos = window["Intensity"].max()
            si_labels.append(
                {"label": p["hkl"], "x": search_th, "y": y_pos, "color": "red"}
            )

# ICDDデータのラベル
ax2_labels = []
for _, row in ref_df.iterrows():
    if row["I(f)"] >= 5 and row["2Theta(deg)"] <= 90:  # 強度5以上かつ90°以下
        ax2_labels.append(
            {
                "label": f"({int(row['h'])}{int(row['k'])}{int(row['l'])})",
                "x": row["2Theta(deg)"],
                "y": row["I(f)"],
                "color": "blue",
            }
        )

# ラベルの重なり処理
unique_labels_ax1 = get_unique_labels(bt_labels) + get_unique_labels(
    si_labels
)  # BaTiO3とSiのラベルをそれぞれ処理して、一つのリストに格納
# unique_labels_ax1 = get_unique_labels(
#     bt_labels
# )  # BaTiO3とSiのラベルをそれぞれ処理して、一つのリストに格納
unique_labels_ax2 = get_unique_labels(ax2_labels)

# 5. プロット
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)

# ax1への描画
ax1.plot(df["2theta_corr"], df["Intensity"], "k-", linewidth=0.8, label="Corrected XRD")

# ax1のラベルの描画
for l in unique_labels_ax1:
    ax1.text(
        l["x"],
        l["y"] + 500,
        l["label"],
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=8,
        color=l["color"],
    )

ax1.plot([], [], "o", color="blue", label="BaTiO3 Peaks", linestyle="None")
ax1.plot([], [], "o", color="red", label="Si Peaks", linestyle="None")

ax1.legend()
# ax1.set_xlabel(r"2$\theta$ (degrees)")
ax1.set_ylabel("Intensity (a.u.)")
ax1.set_ylim(0, df["Intensity"].max() * 1.3)  # タイトルにも結果を表示

# ax2への描画
ax2.vlines(
    ref_df["2Theta(deg)"],
    0,
    ref_df["I(f)"],
    colors="black",
    linewidth=1.2,
    label="BaTiO3 (Tetragonal)",
)

# 各スティックの上に指数のラベルを表示
for l in unique_labels_ax2:
    ax2.text(
        l["x"],
        l["y"] + 5,
        l["label"],
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=7,
        color=l["color"],
    )

# ax2 の書式設定
ax2.set_ylabel("Rel. Int. (ICDD)")
ax2.set_ylim(0, 180)  # ラベルが見切れないように高さを調整
ax2.legend(loc="upper right")
ax2.grid(
    axis="x", linestyle="--", alpha=0.5
)  # ピーク位置を ax1 と比較しやすくするために縦グリッドを追加

# 共通のX軸設定 (一番下の ax2 に設定)
ax2.set_xlabel(r"2$\theta$ (degrees)")
ax2.set_xlim(20, 90)  # プロット範囲を少し広げました

plt.tight_layout()
plt.savefig(f"basic/out/{sample_name}.png")
# plt.show()
plt.close()
