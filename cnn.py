import itertools
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, MaxPooling1D, Input, Flatten, Dense, Activation, Concatenate, Masking, Dropout
from keras.activations import swish, tanh, relu
from tensorflow.python.keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, f1_score
import sklearn.metrics as skm


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


class PlotLosses(keras.callbacks.Callback):
    def __init__(self, val_data=None, train_data=None):
        super().__init__()
        self.validation_data = val_data
        self.train_data = train_data
        plt.ion()
        self.fig = plt.figure()

    def on_train_begin(self, logs={}):
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.f1 = []
        self.val_f1 = []
        self.val_f1_1 = []
        self.logs = []
        self.legend = False
        self.best_acc = 0
        self.best_f1_0 = 0
        self.best_f1_1 = 0
        self.best_pr0 = 0
        self.best_pr1 = 0
        self.best_loss = 1
        self.val_pr0 = []
        self.val_pr1 = []
        self.val_rc0 = []
        self.val_rc1 = []
        self.pr0 = []
        self.pr1 = []
        self.rc0 = []
        self.rc1 = []

    def on_epoch_end(self, epoch, logs={}):
        self.x.append(epoch)
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('categorical_accuracy'))
        self.val_acc.append(logs.get('val_categorical_accuracy'))
        self.f1.append(logs.get('f1_score'))
        self.val_f1.append(logs.get('val_f1_score'))


        if epoch % 10 == 0:
            plt.close()
            self.legend = False
            return

        plt.subplot(1, 3, 1)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.losses, 'b', label="loss")
        plt.plot(self.x, self.val_losses, 'r', label="val_loss")
        if not self.legend:
            plt.legend(loc='lower left')
        plt.subplot(1, 3, 2)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.f1, 'b', label="f1")
        plt.plot(self.x, self.val_f1, 'r', label="val_f1")
        plt.ylim([0, 1])
        if not self.legend:
            plt.legend(loc='lower left')
        plt.subplot(1, 3, 3)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.acc, 'b', label="acc")
        plt.plot(self.x, self.val_acc, 'r', label="val_acc")
        plt.ylim([0, 1])
        if self.legend == False:
            plt.legend(loc='lower left')
            self.legend = True
        plt.pause(0.01)


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
            "check1_value", "check2_value",
            "check3_value", "check4_value", "check5_value", "check6_value",
        ]
        self.feature_nums = len(self.features)

        self.preprocessing()
    
    def preprocessing(self):
        raw_data1_df = pd.read_csv("raw_data1.csv")
        raw_data2_df = pd.read_csv("raw_data2.csv")
        raw_data1_df = raw_data1_df.sort_values(by="reg_date").copy()
        raw_data2_df = raw_data2_df.sort_values(by="reg_date").copy()
        self.id_list = np.unique(raw_data1_df["menti_seq"]).tolist()

        self.np_x = np.ones((len(self.id_list), self.TIME_SLOT, self.feature_nums), dtype=np.int32) * -1  # Specify shape based on your data
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

                if len(y_tmp) > 1:  # abnomal condition
                    self.np_y[idx, d_idx] = np.clip(y_tmp[0] * y_tmp[-1], None, 1)
                    self.np_x_sup[idx, d_idx, -len(y_tmp) + 1:] = np.clip(y_tmp[:-1], None, 1)

    def create_model(self):
        print("[COCOModel] create_model()")
        list_concat = []
        list_input = []
        for i in range(2):
            input_name = "input_" + str(i)
            if i == 0:
                input = Input(shape=(40, self.feature_nums), name=input_name, batch_size=self.BATCH_SIZE)
            else:
                pass

            list_input.append(input)

            input = Masking(mask_value=0)(input) #masking 40개의 설문이없는경우 0으로설정
            conv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(input)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            list_concat.append(conv)

        z = Concatenate(axis=1)(list_concat)
        z = Dense(128, activation='relu')(z)
        z = Dropout(0.1)(z) #dropout 비율
        output = Dense(2, activation='softmax')(z)

        model = keras.Model(inputs=list_input[0], outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=[keras.metrics.CategoricalAccuracy(),
                               keras.metrics.F1Score(average="macro")
                               ]
                      )
        model.summary()
        return model

    def make_ds(self, d):
        d_idx = self.D.index(d)
        mask = np.ones((len(self.id_list), 1), dtype=np.uint8)
        for id in self.id_list:
            idx = self.id_list.index(id)
            if self.np_y[idx, d_idx] == -1:
                mask[idx] = 0
                continue
        self.np_x = self.np_x[mask[:, 0] == 1]
        self.np_x_sup = self.np_x_sup[mask[:, 0] == 1]
        self.np_y = self.np_y[mask[:, 0] == 1]
        self.np_yy = to_categorical(self.np_y[:, d_idx], num_classes=len(self.CLASS))

        total = len(self.np_y[:, 0])
        cnt_0 = len(np.where(self.np_y[:, 0] == 0)[0]) / total
        cnt_1 = len(np.where(self.np_y[:, 0] == 1)[0]) / total


        print("\tnormal: %.2f\tabnormal: %.2f" % (cnt_0, cnt_1,))

        dict_train = {}
        for i in range(2):
            if i == 0:
                tf_tmp = tf.data.Dataset.from_tensor_slices(self.np_x)
            else:
                tf_tmp = tf.data.Dataset.from_tensor_slices(self.np_x_sup[:, d_idx, :].reshape((self.np_x_sup.shape[0], -1, 1)))

            tf_name = "input_" + str(i)
            dict_train[tf_name] = tf_tmp

        labels = tf.data.Dataset.from_tensor_slices(self.np_yy)
        dataset = tf.data.Dataset.zip((dict_train, labels))

        data_size = len(self.np_y[:, 1])
        self.train_size = int(data_size * 0.8) #test case 1-datasize
        self.test_size = data_size - self.train_size
        self.train_dataset = dataset.take(self.train_size)
        self.test_dataset = dataset.skip(self.train_size)

        self.train_dataset = self.train_dataset.shuffle(self.train_size).batch(self.BATCH_SIZE)
        self.test_dataset = self.test_dataset.batch(self.BATCH_SIZE)

        self.train_dataset = self.train_dataset.cache()
        self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        self.test_dataset = self.test_dataset.cache()
        self.test_dataset = self.test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def train(self):
        self.make_ds("PHQ-9")
        self.model = self.create_model()
        try:
            self.model.fit(
                x=self.train_dataset,
                validation_data=self.test_dataset,
                epochs=500,
                callbacks=[PlotLosses()]
            )
        except KeyboardInterrupt:
            pass

    def test(self):
        target_y = self.np_yy[-self.test_size:]
        pred = self.model.predict(self.test_dataset)
        pred = to_categorical(tf.argmax(pred, axis=1), num_classes=2)
        y_pred = np.argmax(pred, axis=1)
        y_true = np.argmax(target_y, axis=1)
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        print(skm.classification_report(y_true, y_pred))
        plot_confusion_matrix(cm=cm, classes=self.CLASS, title='confusion_matrix')


m = Mental()

m.train()
m.test()
