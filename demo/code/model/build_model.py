#from keras_self_attention import SeqSelfAttention
from Muattention import *
import keras
import keras.callbacks
from keras.layers import Conv1D, Activation, Dropout,regularizers
import keras.backend


def build_model(model_template, number_of_kernel, kernel_length, input_shape, local_window_size=19):

    '''
      def bias_variable(shape):
          initial = tf.constant(0.1, shape=shape)
          initial=tf.Variable(initial)
          return initial
    '''

    def Relu(x):
        return keras.backend.relu(x, alpha=0.5, max_value=10)

    model_template.add(Conv1D(
        input_shape=input_shape,
        kernel_size=kernel_length,
        filters=number_of_kernel,
        padding='valid',
        strides=1))


    model_template.add(Activation(Relu))
    model_template.add(keras.layers.pooling.MaxPooling1D(pool_length=local_window_size,
                                                         stride=None, border_mode='valid'))

    model_template.add(Dropout(0.5))

    model_template.add(keras.layers.GRU(
        output_dim=number_of_kernel,
        return_sequences=True))
    model_template.add(Activation(Relu))

    model_template.add(Multi_Head_attention(out_dim=number_of_kernel))

    model_template.add(keras.layers.GlobalMaxPooling1D())
    model_template.add(Dropout(0.5))
    model_template.add(keras.layers.core.Dense(output_dim=1, name='Dense_l1'))
    model_template.add(keras.layers.Activation("sigmoid"))
    sgd = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005, amsgrad=False)
    return model_template, sgd



if __name__ == '__main__':

    pass
