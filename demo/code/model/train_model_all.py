import sys
from build_model import *
from data_load2 import *
import os
import numpy as np
import pickle
import random
import tensorflow as tf
import keras
from sklearn.model_selection import StratifiedKFold
import numpy
from math import *
import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_auc_score, auc, roc_curve


plt.switch_backend('agg')


def mkdir(path):
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)
        return True
    else:
        return False


def AUC(label, pred):
    roc_auc_score(label, pred)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(label, pred)
    roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)
    return roc_auc


def  train_Hybrid_model(number_of_kernel,
                    ker_len, input_shape,
                    batch_size,
                    epoch_num,
                    data_info,
                    modelsave_output_prefix,
                    random_seed,
                    local_window_size=8):

    model = keras.models.Sequential()

    model, sgd = build_model(model,
                           number_of_kernel,
                           ker_len,
                           input_shape=input_shape,
                           local_window_size=local_window_size)

    model.compile(loss='binary_crossentropy', optimizer=sgd)

    # set the result path
    output_path = modelsave_output_prefix + "\\" + str(data_info)  # ....
    mkdir(output_path)
    output_prefix = output_path + "/" \
                    + "model-KernelNum_" + str(number_of_kernel) \
                    + "_random-seed_" + str(random_seed) \
                    + "_batch-size_" + str(batch_size) \
                    + '_kernel-length_' + str(ker_len) \
                    + '_localwindow_' + str(local_window_size)

    modelsave_output_filename = output_prefix + "_checkpointer.hdf5"
    history_output_path = output_prefix + '.history'
    prediction_save_path = output_prefix + '.npy'

    if os.path.exists(prediction_save_path):
        print('file has existed')
        return 0, 0
    # train the model, and save the history

    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=8,
                                                 verbose=0)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=modelsave_output_filename,
                                                   verbose=0,
                                                   save_best_only=True)

    history = model.fit(X_train,
                        Y_train,
                        epochs=epoch_num,
                        batch_size=batch_size,
                        validation_data=(X_val,Y_val),
                        verbose=1,
                        callbacks=[checkpointer, earlystopper]
                        )

    # load the best weight
    #model.save_weights(modelsave_output_filename)
    model.load_weights(modelsave_output_filename)

    # get the prediction of the test data set, and save as .npy

    prediction = model.predict(X_test)
    auc= AUC(Y_test, prediction)
    cvscores.append(auc)
    print(auc)

    #prediction = numpy.mean(cvscores)
    #np.save(prediction_save_path, auc)

    c=len(cvscores)
    if c%10==0:
        AUCs= numpy.mean(cvscores)
        np.save(prediction_save_path, AUCs)
        print(c)
        print(AUCs)
        cvscores.clear()
        #AUCS = numpy.mean(cvscores)



    # save the history
    with open(history_output_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return history, auc


# read the hyper-parameters
data_path = sys.argv[1]
result_path = sys.argv[2]
data_info = sys.argv[3]
number_of_kernel = int(sys.argv[4])
random_seed = int(sys.argv[5])
local_window = int(sys.argv[6])
GPU_SET = sys.argv[7]

# set the hyper-parameters
ker_len = 15
batch_size = 32
epoch_num = 100

np.random.seed(random_seed)
random.seed(random_seed)
tf.set_random_seed(random_seed)
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_SET
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#10-fold-valid
X , Y = get_data(data_path)
idx = X.shape[0]
input_shape = (X.shape[1],X.shape[2])
cvscores = []
for i in range(10):
    print(i, "times: ")
    print(input_shape, idx, '==========')
    X_test =X[int(idx * i * 0.1):int(idx * (i + 1) * 0.1), ]
    Y_test = Y[int(idx * i * 0.1):int(idx * (i + 1) * 0.1), ]
    if i + 1 <= max(range(10)):
        X_val = X[int(idx * (i + 1) * 0.1):int((i + 2) * idx * 0.1)]
        Y_val = Y [int(idx * (i + 1) * 0.1):int((i + 2) * idx * 0.1)]

        X_train = np.delete(X , range(int(idx * i * 0.1), int(idx * (i + 2) * 0.1)), axis=0)
        Y_train = np.delete(Y, range(int(idx * i * 0.1), int(idx * (i + 2) * 0.1)), axis=0)

    else:
        X_val =X[:int(((i + 1) % 8) * idx * 0.1)]
        Y_val = Y[:int(((i + 1) % 8) * idx * 0.1)]

        X_train = np.delete(X, range(int(idx * i * 0.1), int(idx * (i + 1) * 0.1)), axis=0)
        X_train = np.delete(X_train, range(int(((i + 1) % 8) * idx * 0.1)), axis=0)

        Y_train= np.delete(Y, range(int(idx * i * 0.1), int(idx * (i + 1) * 0.1)), axis=0)
        Y_train = np.delete(Y_train, range(int(((i + 1) % 8) * idx * 0.1)), axis=0)

    History_Soft, prediction_Soft =  train_Hybrid_model(number_of_kernel=number_of_kernel,
                                                    ker_len=ker_len,
                                                    input_shape=input_shape,
                                                    batch_size=batch_size,
                                                    epoch_num=epoch_num,
                                                    data_info=data_info,
                                                    modelsave_output_prefix=result_path,
                                                    random_seed=random_seed,
                                                    local_window_size=local_window)









