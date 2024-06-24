import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.utils.np_utils import to_categorical


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

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
        self.TIME_SLOT = 40
        self.CLASS = ["normal", "abnormal"]
        self.BATCH_SIZE = 1024

        self.features = [
            "check1_value", #"check2_value",
            "check3_value", "check4_value", "check5_value", 
            #"check6_value",
            "service1", "service2", "service3",
            "service9", "service10",
        ]
        self.feature_nums = len(self.features)

        self.preprocessing()

    def preprocessing(self):
        raw_data1_df = pd.read_csv("raw_data1.csv")
        raw_data2_df = pd.read_csv("raw_data2.csv")
        raw_data1_df = raw_data1_df.sort_values(by="reg_date").copy()
        raw_data2_df = raw_data2_df.sort_values(by="reg_date").copy()
        self.id_list = np.unique(raw_data1_df["menti_seq"]).tolist()

        self.np_x = np.ones((len(self.id_list), self.feature_nums), dtype=np.float32) * -1
        self.np_y = np.ones((len(self.id_list), 3), dtype=np.int32) * -1
        self.np_x_sup = np.ones((len(self.id_list), 3, 130), dtype=np.int32) * -1

        for id in self.id_list:
            xx = raw_data1_df.loc[raw_data1_df["menti_seq"] == id].copy()
            idx = self.id_list.index(id)
            np_data = xx[self.features].to_numpy()

            # Average the check values for each feature over available time slots
            avg_data = np.mean(np_data, axis=0)
            self.np_x[idx] = avg_data

            for d_idx, d_row in enumerate(self.D_LIST):
                d = d_row["name"]
                y_tmp = raw_data2_df.loc[
                    (raw_data2_df["menti_seq"] == id) & (raw_data2_df["srvy_name"] == d), "srvy_result"].to_numpy()

                if len(y_tmp) > 1:
                    clipped_y_tmp = np.clip(y_tmp, 0, 1)
                    mean_y_tmp = np.mean(clipped_y_tmp)
                    if mean_y_tmp >= 0.6:
                        self.np_y[idx, d_idx] = 1
                    else:
                        self.np_y[idx, d_idx] = 0
                    self.np_x_sup[idx, d_idx, -len(y_tmp) + 1:] = np.clip(y_tmp[:-1], None, 1)

    def create_model(self, C_value):
        model = SVC(kernel='linear', C=C_value, probability=True)
        return model

    def make_ds(self, d):
        d_idx = self.D.index(d)
        mask = np.ones((len(self.id_list), 1), dtype=np.uint8)
        for id in self.id_list:
            idx = self.id_list.index(id)
            if self.np_y[idx, d_idx] == -1:
                mask[idx] = 0

        self.np_x = self.np_x[mask[:, 0] == 1]
        self.np_x_sup = self.np_x_sup[mask[:, 0] == 1]
        self.np_y = self.np_y[mask[:, 0] == 1]
        self.np_yy = to_categorical(self.np_y[:, d_idx], num_classes=len(self.CLASS))

        total = len(self.np_y[:, 0])
        cnt_0 = len(np.where(self.np_y[:, 0] == 0)[0]) / total
        cnt_1 = len(np.where(self.np_y[:, 0] == 1)[0]) / total
        cnt_2 = len(np.where(self.np_y[:, 0] == -1)[0]) / total

        print("\tnormal: %.2f\tabnormal: %.2f, %.2f" % (cnt_0, cnt_1, cnt_2))

        # Standardizing the features
        scaler = StandardScaler()
        self.np_x = scaler.fit_transform(self.np_x)

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.np_x, self.np_yy, test_size=0.1, random_state=42)

    def train(self, C_value=1.0):
        self.make_ds("PHQ-9")
        self.model = self.create_model(C_value)
        self.model.fit(self.train_x, np.argmax(self.train_y, axis=1))

    def test(self):
        pred = self.model.predict(self.test_x)
        y_true = np.argmax(self.test_y, axis=1)
        cm = confusion_matrix(y_true=y_true, y_pred=pred)
        print(classification_report(y_true, pred))
        plot_confusion_matrix(cm=cm, classes=self.CLASS, title='confusion_matrix')


m = Mental()
m.train(C_value=1.0)
m.test()
