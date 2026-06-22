import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ========= 설정 =========
INPUT_CSV = "mushra.csv"
SUMMARY_CSV = "mushra_summary.csv"
PLOT_PNG = "mushra_dotplot_ci95.png"

MODEL_COLOR = "tab:blue"   # 성능 비교 모델들 색
REF_COLOR   = "tab:orange" # 앵커/레퍼런스 색
MARKER_SIZE = 6

# ========= 1. 데이터 로드 =========
df = pd.read_csv(INPUT_CSV)
# df = df[df["email"] == "jessica7050@naver.com"]

# score 숫자형 변환
df["rating_score"] = pd.to_numeric(df["rating_score"], errors="coerce")
df["email"]

# ========= 2. 라벨 매핑 =========
# stimulus → 사람이 읽을 수 있는 이름
label_map = {
    "C1": "SAT (4.6kbps)",
    "C2": "SNAC (2.6kbps)",
    "C3": "WNAC w/o waveloss \n (5.2kbps)",
    "C4": "WNAC (5.2kbps)",
    "C5": "upscale (5.2kbps)",
    "C6": "downscale (5.2kbps)",
    "C7": "WNAC (2.52kbps)",   # 필요 없으면 지워도 됨
    "reference": "Reference",
    "anchor70": "Anchor 70",
    "anchor35": "Anchor 30",
}

df["Condition"] = df["rating_stimulus"].map(label_map).fillna(df["rating_stimulus"])

# ========= 3. 요약 통계 (mean, std, count, CI95) =========
summary = df.groupby("Condition")["rating_score"].agg(["mean", "std", "count"]).sort_index()
summary["sem"] = summary["std"] / np.sqrt(summary["count"])
summary["ci95"] = 1.96 * summary["sem"]

# Condition별 요약을 CSV로 저장
summary.to_csv(SUMMARY_CSV)

# ========= 4. 플롯에 쓸 순서 지정 =========
# 실험에 포함한 조건만 순서대로 나열
order = [
    "Reference",
    "Anchor 30",
    "Anchor 70",
    "SAT (4.6kbps)",
    "SNAC (2.6kbps)",
    "downscale (5.2kbps)",
    "upscale (5.2kbps)",
    "WNAC w/o waveloss \n (5.2kbps)",
    "WNAC (2.52kbps)",
    "WNAC (5.2kbps)",
]

# 실제 summary에 존재하는 조건만 필터링 (없는 이름 있으면 자동으로 제거)
order_in_data = [cond for cond in order if cond in summary.index]
summary = summary.loc[order_in_data]

# ========= 5. 그룹별 색 지정 =========
performance_models = {"SAT (4.6kbps)", "SNAC (2.6kbps)", "downscale (5.2kbps)", "upscale (5.2kbps)", "WNAC w/o waveloss \n (5.2kbps)", "WNAC (2.52kbps)", "WNAC (5.2kbps)"}
colors = [MODEL_COLOR if cond in performance_models else REF_COLOR for cond in summary.index]

# ========= 6. 플롯 =========
fig, ax = plt.subplots(figsize=(7.5, 6))
x = np.arange(len(summary))

# 조건별로 색을 달리한 에러바(95% CI) + 마커
for i, (cond, row) in enumerate(summary.iterrows()):
    ax.errorbar(
        x=i,
        y=row["mean"],
        yerr=row["ci95"],
        fmt='o',
        markersize=MARKER_SIZE,
        mfc=colors[i],
        mec=colors[i],
        ecolor=colors[i],
        elinewidth=1.4,
        capsize=4,
    )

# 수평 가이드라인
bands = [20, 40, 60, 80]
for y in bands:
    ax.axhline(y, linestyle='--', linewidth=0.7, color='gray')

# 오른쪽 등급 라벨
grade_positions = [10, 30, 50, 70, 90]
grade_labels = ["Bad", "Poor", "Fair", "Good", "Excellent"]
for yp, lab in zip(grade_positions, grade_labels):
    ax.text(1.04, yp, lab, va='center', ha='left', transform=ax.get_yaxis_transform())

# 상단 오른쪽 CI 설명
ax.text(
    0.99, 0.98,
    "95% confidence intervals",
    transform=ax.transAxes,
    ha='right',
    va='top',
    fontsize=10,
)

# 범례
legend_elems = [
    Patch(facecolor=MODEL_COLOR, edgecolor='black', label='Models'),
    Patch(facecolor=REF_COLOR, edgecolor='black', label='Anchors/Reference'),
]
ax.legend(handles=legend_elems, loc='lower right')

# 여백 조정: 오른쪽에 라벨 영역 확보
plt.subplots_adjust(right=0.82)

# 축/제목
ax.set_ylim(0, 100)
ax.set_xticks(x, summary.index, rotation=90)

# 피험자 수: session_uuid 기준으로 추정 (원하면 email로 바꿔도 됨)
if "session_uuid" in df.columns:
    N = df["session_uuid"].nunique()
elif "email" in df.columns:
    N = df["email"].nunique()
else:
    N = int(summary["count"].max() / len(order_in_data))  # fallback

ax.set_title(f"Basic Audio Quality (N={N})")
ax.set_ylabel("Rating (MUSHRA 0–100)")
ax.set_xlabel("")

plt.tight_layout()
plt.savefig(PLOT_PNG, dpi=200)

print(f"요약 통계 저장: {SUMMARY_CSV}")
print(f"플롯 저장: {PLOT_PNG}")
print(summary)


# ========= 7. 사용자(이메일)별 유사도 분석 (모델만 사용) =========
USER_ANALYSIS_CSV = "mushra_user_similarity.csv"

# 어떤 컬럼을 사용자 ID로 쓸지 결정 (우선 email, 없으면 session_uuid)
if "email" in df.columns:
    user_col = "email"
elif "session_uuid" in df.columns:
    user_col = "session_uuid"
else:
    user_col = None

if user_col is not None:
    # ---- (중요) 상관계수 계산에는 모델 조건만 사용 ----
    model_conditions = {
        "SAT (4.6kbps)",
        "SNAC (2.6kbps)",
        "downscale (5.2kbps)",
        "upscale (5.2kbps)",
        "WNAC w/o waveloss \n (5.2kbps)",
        "WNAC (2.52kbps)",
        "WNAC (5.2kbps)",
    }
    # 실제 summary/index에 있는 모델 조건만 사용
    model_conditions_in_summary = [
        c for c in summary.index if c in model_conditions
    ]

    # 원본 df에서 '모델 조건'에 해당하는 평가만 뽑기 (Reference/Anchors 제외)
    df_used = df[df["Condition"].isin(model_conditions_in_summary)].copy()

    # 사용자-조건별 평균 점수 (사용자가 같은 조건을 여러 번 평가했으면 평균)
    user_cond = (
        df_used
        .groupby([user_col, "Condition"])["rating_score"]
        .mean()
        .unstack("Condition")
    )

    # summary에서 "모델 조건"의 전체 평균 벡터
    # (user_cond의 컬럼 순서에 맞춰 reindex)
    global_mean = summary["mean"].reindex(user_cond.columns)

    # 사용자별로 "전체 평균"과의 유사도/차이를 계산
    def calc_similarity(row):
        # 공통으로 평가한 조건만 사용 (NaN 제외)
        mask = row.notna() & global_mean.notna()
        if mask.sum() >= 2:
            # 피어슨 상관계수 (패턴 유사도)
            corr = np.corrcoef(row[mask], global_mean[mask])[0, 1]
        else:
            corr = np.nan

        diff = row - global_mean
        mse = np.nanmean((diff[mask]) ** 2)
        rmse = np.sqrt(mse) if mse == mse else np.nan  # NaN 체크

        # 전체 평균 대비 편향(양수면 전체보다 높게 주는 경향)
        bias = np.nanmean(diff[mask])

        # 사용자 자체 평균 점수 (모델 조건에 대해서만)
        user_mean = np.nanmean(row[mask])

        return pd.Series({
            "corr_with_global": corr,
            "rmse_vs_global": rmse,
            "bias_vs_global": bias,
            "user_mean_score": user_mean,
            "num_conditions_rated": int(mask.sum()),  # 평가한 모델 개수
        })

    user_similarity = user_cond.apply(calc_similarity, axis=1)

    # 상관계수 기준으로 정렬 (가장 "집단 평균에 가까운" 사람 순)
    user_similarity_sorted = user_similarity.sort_values(
        by="corr_with_global", ascending=False
    )

    # CSV로 저장
    user_similarity_sorted.to_csv(USER_ANALYSIS_CSV)
    print(f"사용자별 유사도 분석 저장: {USER_ANALYSIS_CSV}")
    print(user_similarity_sorted.head())

    # ========= 8. 사용자별 유사도 플롯 (별도 그림, 모델 기반) =========
    USER_PLOT_PNG = "mushra_user_similarity_plot.png"

    # corr_with_global 이 NaN이 아닌 사용자만 사용
    valid_users = user_similarity_sorted.dropna(subset=["corr_with_global"])
    if len(valid_users) > 0:
        TOP_K = min(20, len(valid_users))  # 상위 몇 명만 보기
        plot_df = valid_users.head(TOP_K)

        # bias에 따라 색 다르게: 후한 사람(양수) / 짠 사람(음수)
        bar_colors = [
            "tab:green" if b >= 0 else "tab:red"
            for b in plot_df["bias_vs_global"]
        ]

        fig, ax = plt.subplots(figsize=(8, 6))
        y = np.arange(len(plot_df))

        ax.barh(
            y,
            plot_df["corr_with_global"],
            color=bar_colors,
            edgecolor="black",
        )

        # y축에 사용자 ID (email 또는 session_uuid)
        labels = [str(idx) for idx in plot_df.index]
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # 상위 corr이 위로 오게

        ax.set_xlim(-0.1, 1.05)
        ax.set_xlabel("Correlation with global mean (models only)")
        ax.set_title(
            f"User similarity to global MUSHRA pattern (models only)\n"
            f"(top {TOP_K} users by correlation)"
        )

        # 각 바 오른쪽에 bias 표시
        for i, (corr, bias) in enumerate(
            zip(plot_df["corr_with_global"], plot_df["bias_vs_global"])
        ):
            ax.text(
                corr + 0.02,
                i,
                f"bias={bias:+.1f}",
                va="center",
                fontsize=8,
            )

        # 범례: 후한/짠 사람
        legend_elems_user = [
            Patch(facecolor="tab:green", edgecolor="black", label="Generous (bias ≥ 0)"),
            Patch(facecolor="tab:red", edgecolor="black", label="Strict (bias < 0)"),
        ]
        ax.legend(handles=legend_elems_user, loc="lower right")

        plt.tight_layout()
        plt.savefig(USER_PLOT_PNG, dpi=200)
        print(f"사용자 유사도 플롯 저장: {USER_PLOT_PNG}")
    else:
        print("유효한 corr_with_global 값이 없어 사용자 플롯을 건너뜁니다.")

else:
    print("email / session_uuid 컬럼이 없어 사용자별 분석을 건너뜹니다.")