import sys
from build_model import *
from data_load1 import *
import os
import numpy as np
import pickle
import random
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


plt.switch_backend('agg')


def mkdir(path):
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)
        return True
    else:
        return False

def  train_Hybrid_model(number_of_kernel,
                    ker_len, input_shape,
                    batch_size,
                    epoch_num,
                    data_info,
                    modelsave_output_prefix,
                    random_seed,
                    local_window_size = 8):

    model = keras.models.Sequential()
    model, sgd = build_model(model,
                           number_of_kernel,
                           ker_len,
                           input_shape=input_shape,
                           local_window_size= local_window_size)

    model.compile(loss='binary_crossentropy', optimizer=sgd)

    # set the result path
    output_path = modelsave_output_prefix + "\\" + str(data_info)     #....
    mkdir(output_path)
    output_prefix = output_path + "/" \
                                + "model-KernelNum_" + str(number_of_kernel) \
                                + "_random-seed_" + str(random_seed) \
                                + "_batch-size_" + str(batch_size) \
                                + '_kernel-length_' + str(ker_len) \
                                + '_localwindow_'+str(local_window_size)

    modelsave_output_filename = output_prefix + "_checkpointer.hdf5"
    history_output_path = output_prefix + '.history'
    prediction_save_path = output_prefix + '.npy'

    # set the checkpoint and earlystop to save the best model
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 #min_delta=0.0001,
                                                 patience=8,
                                                 verbose=0)

    checkpointer = keras.callbacks.ModelCheckpoint(filepath=modelsave_output_filename,
                                                   verbose=0,
                                                   save_best_only=True)

    if os.path.exists(prediction_save_path):
        print('file has existed')
        return 0, 0
    # train the model, and save the history
    history = model.fit(X_train,
                        Y_train,
                        epochs=epoch_num,
                        batch_size=batch_size,
                        validation_split=0.1,
                        #shuffle=True,
                        verbose=1,
                        callbacks=[checkpointer, earlystopper])


    # load the best weight
    model.load_weights(modelsave_output_filename)

    # get the prediction of the test data set, and save as .npy
    prediction = model.predict(X_test)
    np.save(prediction_save_path, prediction)

    # save the history
    with open(history_output_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return history, prediction


# read the hyper-parameters
data_path = sys.argv[1]
result_path = sys.argv[2]
data_info = sys.argv[3]
number_of_kernel = int(sys.argv[4])
random_seed = int(sys.argv[5])
local_window = int(sys.argv[6])
GPU_SET = sys.argv[7]



np.random.seed(random_seed)
random.seed(random_seed)
tf.set_random_seed(random_seed)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_SET
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X_test, Y_test, X_train, Y_train = get_data(data_path)
input_shape= (X_train.shape[1],X_train.shape[2] )


print(X_train.shape[1])
print(number_of_kernel)

# set the hyper-parameters
batch_size = 32
epoch_num = 100
#ker_len_list=[15]
ker_len_list=[15]
for ker_len in ker_len_list:

    History_Soft, prediction_Soft = train_Hybrid_model(number_of_kernel=number_of_kernel,
                                                    ker_len=ker_len,
                                                    input_shape=input_shape,
                                                    batch_size=batch_size,
                                                    epoch_num=epoch_num,
                                                    data_info=data_info,
                                                    modelsave_output_prefix=result_path,
                                                    random_seed=random_seed,
                                                    local_window_size=local_window)




