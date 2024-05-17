import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties



# 파일 경로 설정
file1_path = "phq_results.csv"  # menti_seq, Predicted, Actual 파일 경로
file2_path = "survey_val_counts.csv"  # menti_seq, check1_val_0, ..., check6_val_3 파일 경로

# 데이터 불러오기
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# 두 데이터프레임 합치기
merged_df = pd.merge(df1, df2, on="menti_seq")

# 실제 값을 기준으로 정상과 비정상(PHQ-9) 그룹으로 나누기
normal_df = merged_df[merged_df['Actual'] == 0]
phq9_df = merged_df[merged_df['Actual'] == 1]

# 각 그룹별 평균 계산
normal_means = normal_df.mean(numeric_only=True)
phq9_means = phq9_df.mean(numeric_only=True)

# 필요한 컬럼만 선택하여 새로운 데이터프레임 생성
check_columns = [col for col in merged_df.columns if 'check' in col]

# 항목 이름 매핑
column_names_mapping = {
    "check1_val_0": "Feeling: Relaxing",
    "check1_val_1": "Feeling: Inconvenient",
    "check1_val_2": "Feeling: depressed",
    "check2_val_0": "Suicide thoughts: none",
    "check2_val_1": "Suicide thoughts: yes",
    "check3_val_0": "Anxiety: None",
    "check3_val_1": "Anxiety: Has",
    "check4_val_0": "Dine: Normal",
    "check4_val_1": "Dine: Less",
    "check5_val_1": "one meal",
    "check5_val_2": "2 meals",
    "check5_val_3": "3 meals",
    "check6_val_0": "Out x",
    "check6_val_1": "1~3 times a week",
    "check6_val_2": "4~6 times a week",
    "check6_val_3": "Everyday",
}

# 컬럼 이름 변경
check_columns_readable = [column_names_mapping[col] for col in check_columns]
mean_df = pd.DataFrame({
    "항목": check_columns_readable,
    "정상": normal_means[check_columns].values,
    "PHQ-9": phq9_means[check_columns].values
})

# 그래프 생성
x = np.arange(len(mean_df))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))

bars1 = ax.bar(x - width/2, mean_df['정상'], width, label='Normal', color='skyblue')
bars2 = ax.bar(x + width/2, mean_df['PHQ-9'], width, label='PHQ-9', color='coral')

# 제목과 레이블 설정
ax.set_title('Average number of checks')
ax.set_xlabel('category')
ax.set_ylabel('number of checks')
ax.set_xticks(x)
ax.set_xticklabels(mean_df['항목'], rotation=45, ha='right')
ax.legend()

# 바 라벨 추가
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()
plt.show()
