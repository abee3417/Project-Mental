import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from matplotlib import pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.ioff()
    plt.close()
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

class Mental():
    def __init__(self):
        self.D = ["PHQ-9", "P4", "Loneliness"]
        self.D_LIST = [
            {"name": "PHQ-9", "win": 6},
            {"name": "P4", "win": 3},
            {"name": "Loneliness", "win": 3}
        ]
        self.TIME_SLOT = 40  # 설문조사 갯수
        self.CLASS = ["normal", "abnormal"]
        self.BATCH_SIZE = 1024

        self.features = [
            "check1_value", "check2_value",
            "check3_value", "check4_value",
            "check5_value", "check6_value",
        ]
        self.feature_nums = len(self.features)

        self.preprocessing()
    
    def preprocessing(self):
        raw_data1_df = pd.read_csv("raw_data1.csv")
        raw_data2_df = pd.read_csv("raw_data2.csv")
        raw_data1_df = raw_data1_df.sort_values(by="reg_date").copy()
        raw_data2_df = raw_data2_df.sort_values(by="reg_date").copy()
        self.id_list = np.unique(raw_data1_df["menti_seq"]).tolist()

        self.np_x = np.ones((len(self.id_list), self.TIME_SLOT, self.feature_nums), dtype=np.int32) * -1
        self.np_y = np.ones((len(self.id_list), 6), dtype=np.int32) * -1  # 6개의 check value에 맞게 변경
        self.np_x_sup = np.ones((len(self.id_list), 6, 130), dtype=np.int32) * -1  # 6개의 check value에 맞게 변경

        for id in self.id_list:
            xx = raw_data1_df.loc[raw_data1_df["menti_seq"] == id].copy()
            idx = self.id_list.index(id)
            np_data = xx[self.features].to_numpy()
            time_len = len(np_data[:, 1])
            self.feature_nums = len(np_data[0, :])

            self.np_x[idx, -time_len:] = np_data

            for d_idx, d_row in enumerate(self.D_LIST):
                d = d_row["name"]
                y_tmp = raw_data2_df.loc[
                    (raw_data2_df["menti_seq"] == id) & (raw_data2_df["srvy_name"] == d), "srvy_result"].to_numpy()

                if len(y_tmp) > 1:  # abnormal condition
                    clipped_y_tmp = np.clip(y_tmp, 0, 1)  # 1 이상인 값을 1로 클리핑
                    mean_y_tmp = np.mean(clipped_y_tmp)
                    if mean_y_tmp >= 0.3:
                        self.np_y[idx, d_idx] = 1
                    else:
                        self.np_y[idx, d_idx] = 0
                    self.np_x_sup[idx, d_idx, -len(y_tmp) + 1:] = clipped_y_tmp[:-1]

    def create_model(self):
        print("[XGBoost Model] create_model()")

    def make_ds(self, d):
        d_idx = self.D.index(d)
        mask = np.ones((len(self.id_list), 1), dtype=np.uint8)
        for id in self.id_list:
            idx = self.id_list.index(id)
            if self.np_y[idx, d_idx] == -1:
                mask[idx] = 0
                continue
        self.np_x = self.np_x[mask[:, 0] == 1].reshape(-1, self.TIME_SLOT * self.feature_nums)
        self.np_x_sup = self.np_x_sup[mask[:, 0] == 1]
        self.np_y = self.np_y[mask[:, 0] == 1]
        self.np_yy = self.np_y[:, d_idx]

        total = len(self.np_y[:, 0])
        cnt_0 = len(np.where(self.np_y[:, 0] == 0)[0]) / total
        cnt_1 = len(np.where(self.np_y[:, 0] == 1)[0]) / total

        print("\tnormal: %.2f\tabnormal: %.2f" % (cnt_0, cnt_1))

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.np_x, self.np_yy, test_size=0.2, random_state=42)  # test size

    def train(self):
        self.make_ds("P4")
        
        # 오버샘플링 및 언더샘플링 적용
        smote = SMOTE(sampling_strategy=1.0, random_state=42)  # 소수 클래스 비율을 다수 클래스와 같게 증강
        undersample = RandomUnderSampler(sampling_strategy=1.0, random_state=42)  # 다수 클래스 비율을 소수 클래스와 같게 축소
        
        # 오버샘플링
        X_resampled, y_resampled = smote.fit_resample(self.train_x, self.train_y)
        
        # 언더샘플링
        X_resampled, y_resampled = undersample.fit_resample(X_resampled, y_resampled)
        
        self.create_model()
        # 불균형 데이터 가중치 조정
        scale_pos_weight = (len(y_resampled) - sum(y_resampled)) / sum(y_resampled)
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,  # 부분 샘플링 비율을 설정하여 과적합 방지
            colsample_bytree=0.8  # 부분 피처 샘플링 비율을 설정하여 과적합 방지
        )
        self.model.fit(X_resampled, y_resampled)

    def test(self):
        pred = self.model.predict(self.test_x)
        cm = confusion_matrix(self.test_y, pred)
        print(classification_report(self.test_y, pred))
        plot_confusion_matrix(cm=cm, classes=self.CLASS, title='confusion_matrix')

        # menti_seq별 예측 결과 저장
        results = []
        for i, menti_seq in enumerate(self.test_id):
            results.append({
                "menti_seq": menti_seq,
                "Predicted": pred[i],
                "Actual": self.test_y[i]
            })

        # 결과를 파일로 저장
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="menti_seq")
        results_df.to_csv("prediction_results.csv", index=False)

        cm = confusion_matrix(self.test_y, pred)
        print(classification_report(self.test_y, pred))
        plot_confusion_matrix(cm=cm, classes=self.CLASS, title='confusion_matrix')


m = Mental()

m.train()
m.test()
