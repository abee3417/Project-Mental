import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
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
        self.TIME_SLOT = 40  # 설문조사갯수
        self.CLASS = ["normal", "abnormal"]
        self.BATCH_SIZE = 1024

        self.features = [
            "check1_value", "check2_value", 
            "check3_value", "check4_value", "check5_value", "check6_value",
            "service1", "service2", "service3", "service4", "service5", 
            "service6", "service7", "service8", "service9", "service10", 
            "service11", "service12", "service13", "service14", "service15", 
            "service16", 
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
        self.np_y = np.ones((len(self.id_list), 3), dtype=np.int32) * -1
        self.np_x_sup = np.ones((len(self.id_list), 3, 130), dtype=np.int32) * -1

        for id in self.id_list:
            print(id)
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
                    self.np_y[idx, d_idx] = np.clip(y_tmp[0] * y_tmp[-1], None, 1)
                    self.np_x_sup[idx, d_idx, -len(y_tmp) + 1:] = np.clip(y_tmp[:-1], None, 1)

    def create_model(self, scale_pos_weight):
        print("[XGBoost Model] create_model()")
        return xgb.XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            scale_pos_weight=scale_pos_weight
        )

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
        self.id_list_filtered = np.array(self.id_list)[mask[:, 0] == 1]  # Filter id_list

        total = len(self.np_y[:, 0])
        cnt_0 = len(np.where(self.np_y[:, 0] == 0)[0]) / total
        cnt_1 = len(np.where(self.np_y[:, 0] == 1)[0]) / total

        print("\tnormal: %.2f\tabnormal: %.2f" % (cnt_0, cnt_1))

        self.train_x, self.test_x, self.train_y, self.test_y, self.train_id, self.test_id = train_test_split(
            self.np_x, self.np_yy, self.id_list_filtered, test_size=0.1, random_state=42
        )

    def train(self):
        self.make_ds("Loneliness")

        # 언더샘플링 모델
        rus = RandomUnderSampler(random_state=42)
        train_x_rus, train_y_rus = rus.fit_resample(self.train_x, self.train_y)
        self.model_rus = self.create_model(scale_pos_weight=1)
        self.model_rus.fit(train_x_rus, train_y_rus)

        # 오버샘플링 모델
        smote = SMOTE(random_state=42)
        train_x_smote, train_y_smote = smote.fit_resample(self.train_x, self.train_y)
        self.model_smote = self.create_model(scale_pos_weight=1)
        self.model_smote.fit(train_x_smote, train_y_smote)

    def test(self):
        # 두 모델의 예측 결과를 평균하여 결합
        pred_rus = self.model_rus.predict_proba(self.test_x)[:, 1]
        pred_smote = self.model_smote.predict_proba(self.test_x)[:, 1]
        pred = (pred_rus + pred_smote) / 2
        pred_binary = np.where(pred > 0.5, 1, 0)

        # menti_seq별 예측 결과 저장
        results = []
        for i, menti_seq in enumerate(self.test_id):
            results.append({
                "menti_seq": menti_seq,
                "Predicted": pred_binary[i],
                "Actual": self.test_y[i]
            })

        # 결과를 파일로 저장
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="menti_seq")
        results_df.to_csv("prediction_results.csv", index=False)

        cm = confusion_matrix(self.test_y, pred_binary)
        print(classification_report(self.test_y, pred_binary))
        plot_confusion_matrix(cm=cm, classes=self.CLASS, title='confusion_matrix')

m = Mental()
m.train()
m.test()