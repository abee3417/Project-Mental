import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.utils.fixes import threadpool_limits

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
            "service11", "service12", "service13", "service14", "service15", "service16"
        ]
        self.feature_nums = len(self.features)

        self.preprocessing()

    def preprocessing(self):
        raw_data1_df = pd.read_csv("/content/sample_data/raw_data1.csv")
        raw_data2_df = pd.read_csv("/content/sample_data/raw_data2.csv")
        raw_data1_df = raw_data1_df.sort_values(by="reg_date").copy()
        raw_data2_df = raw_data2_df.sort_values(by="reg_date").copy()
        self.id_list = np.unique(raw_data1_df["menti_seq"]).tolist()

        self.np_x = np.ones((len(self.id_list), self.TIME_SLOT, self.feature_nums), dtype=np.int32) * -1
        self.np_y = np.ones((len(self.id_list), 3), dtype=np.int32) * -1
        self.np_x_sup = np.ones((len(self.id_list), 3, 130), dtype=np.int32) * -1

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
                    self.np_y[idx, d_idx] = np.clip(y_tmp[0] * y_tmp[-1], None, 1)
                    self.np_x_sup[idx, d_idx, -len(y_tmp) + 1:] = np.clip(y_tmp[:-1], None, 1)

    def create_model(self):
        print("XGBoost Model")
        return xgb.XGBClassifier(
            n_estimators=1000, 
            scale_pos_weight=1
        )

    def make_ds(self, d):
        d_idx = self.D.index(d)
        mask = np.ones((len(self.id_list), 1), dtype=np.uint8)
        for id in self.id_list:
            idx = self.id_list.index(id)
            if self.np_y[idx, d_idx] == -1:
                mask[idx] = 0
                continue

        # 시간 축으로 평균낸 후, reshape
        self.np_x = self.np_x[mask[:, 0] == 1].mean(axis=1)
        self.np_x_sup = self.np_x_sup[mask[:, 0] == 1]
        self.np_y = self.np_y[mask[:, 0] == 1]
        self.np_yy = self.np_y[:, d_idx]
        self.id_list_filtered = np.array(self.id_list)[mask[:, 0] == 1]  # Filter id_list

        total = len(self.np_y[:, 0])
        cnt_0 = len(np.where(self.np_y[:, 0] == 0)[0]) / total
        cnt_1 = len(np.where(self.np_y[:, 0] == 1)[0]) / total

        print("\tnormal: %.2f\tabnormal: %.2f" % (cnt_0, cnt_1))

        self.train_x, self.test_x, self.train_y, self.test_y, self.train_id, self.test_id = train_test_split(
            self.np_x, self.np_yy, self.id_list_filtered, test_size=0.2, random_state=42
        )

    def train(self):
        self.make_ds("Loneliness")

        # 오버샘플링 모델
        smote = SMOTE(random_state=42)
        train_x_smote, train_y_smote = smote.fit_resample(self.train_x, self.train_y)

        # 하이퍼파라미터 그리드 정의
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }

        # 모델 생성
        model = self.create_model()

        # GridSearchCV 설정
        with threadpool_limits(limits=1, user_api='blas'):
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1', verbose=1, n_jobs=-1)

            # 모델 학습
            grid_search.fit(train_x_smote, train_y_smote)

        # 최적 모델 저장
        self.model_smote = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")

    def test(self):
        # 오버샘플링 모델 예측
        pred_smote = self.model_smote.predict_proba(self.test_x)[:, 1]
        pred_binary = np.where(pred_smote > 0.5, 1, 0)

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
        report = classification_report(self.test_y, pred_binary, output_dict=True, zero_division=0)
        print(classification_report(self.test_y, pred_binary, zero_division=0))

        macro_f1 = report['macro avg']['f1-score']
        print(f"Macro F1-Score: {macro_f1}")

        plot_confusion_matrix(cm=cm, classes=self.CLASS, title='confusion_matrix')

m = Mental()
m.train()
m.test()