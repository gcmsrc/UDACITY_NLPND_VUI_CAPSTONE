from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout, MaxPooling1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, name='rnn', implementation=2)(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn_cnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1, pool_size=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // (stride * pool_size)

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    input_rnn = input_data
    for i in range(recur_layers):
        input_rnn = GRU(units, activation='relu',
            return_sequences=True, name=f'rnn_{i+1}')(input_rnn)
        input_rnn = BatchNormalization(name=f'bn_rnn_{i+1}')(input_rnn)
    # Rename input_rnn
    bn_rnn = input_rnn
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, name='rnn'), name='bidir_rnn')(input_data)
    # Add batch normalization
    bn_bidir_rnn = BatchNormalization()(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def cnn_bidir_rnn(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Custom deep NN for speech recognition
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # Add convolutional layer
    cnn = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_cnn')(cnn)
    # Apply dropout
    conv_1d_drop = Dropout(0.3, name='drop_bn_cnn')(bn_cnn)
    
    # Add RNN layer (LSTM)
    lstm = LSTM(units, activation='tanh',
        return_sequences=True, name='lstm', dropout=.2, recurrent_dropout=.2)
    # Make it bidirectional
    bidir_lstm = Bidirectional(lstm)(conv_1d_drop)
    
    # Add batch normalization
    bn_lstm = BatchNormalization(name='bn_lstm')(bidir_lstm)
    
    # Add Dense Layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_lstm)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_dilated_bidir_rnn(input_dim, filters, kernel_size, dilation,
    conv_border_mode, units, output_dim=29):
    """ Custom deep NN for speech recognition
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # Add convolutional layer
    cnn = Conv1D(filters, kernel_size, 
                     strides=1, 
                     dilation_rate=dilation,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_cnn')(cnn)
    # Apply dropout
    conv_1d_drop = Dropout(0.3, name='drop_bn_cnn')(bn_cnn)
    
    # Add RNN layer (LSTM)
    lstm = LSTM(units, activation='tanh',
        return_sequences=True, name='lstm', dropout=.2, recurrent_dropout=.2)
    # Make it bidirectional
    bidir_lstm = Bidirectional(lstm)(conv_1d_drop)
    
    # Add batch normalization
    bn_lstm = BatchNormalization(name='bn_lstm')(bidir_lstm)
    
    # Add Dense Layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_lstm)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, stride=1, dilation=dilation)
    print(model.summary())
    return model

def cnn_pooled_bidir_rnn(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29, pool_size=2):
    """ Custom deep NN for speech recognition
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # Add convolutional layer
    cnn = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_cnn')(cnn)
    
    # Add pooling
    pooled_cnn = MaxPooling1D(pool_size=pool_size, name="MaxPooling")(bn_cnn)
    
    # Apply dropout
    conv_1d_drop = Dropout(0.3, name='drop_bn_cnn')(pooled_cnn)
    
    # Add RNN layer (LSTM)
    lstm = LSTM(units, activation='tanh',
        return_sequences=True, name='lstm', dropout=.2, recurrent_dropout=.2)
    # Make it bidirectional
    bidir_lstm = Bidirectional(lstm)(conv_1d_drop)
    
    # Add batch normalization
    bn_lstm = BatchNormalization(name='bn_lstm')(bidir_lstm)
    
    # Add Dense Layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_lstm)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, pool_size=pool_size)
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, dilation,
    conv_border_mode, units, output_dim=29, pool_size=2):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # Add two stacked convoluational layer
    cnn_one = Conv1D(filters, kernel_size, 
                     strides=1, 
                     dilation_rate=dilation,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d_1')(input_data)
    # Add batch normalisation
    bn_cnn_one = BatchNormalization(name="bnn_cnn_1")(cnn_one)
    # Add dropout
    bn_cnn_one_drop = Dropout(0.3, name="cnn_1_drop")(bn_cnn_one)
    
    # Add second convolutional layer
    cnn_two = Conv1D(filters, kernel_size, 
                     strides=1,
                     dilation_rate=dilation,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d_2')(bn_cnn_one_drop)
    bn_cnn_two = BatchNormalization(name="bnn_cnn_2")(cnn_two)
    bn_cnn_two_drop = Dropout(0.3, name="cnn_2_drop")(bn_cnn_two)
    
    # Add max pooling layer (at the end of conv layers)
    pooled_cnn = MaxPooling1D(pool_size=pool_size, name="MaxPooling")(bn_cnn_two_drop)
    
    # Add stacked bidirectional LSTM
    rnn_one = GRU(units, activation='relu',
        return_sequences=True, name='rnn_one', dropout=.2, recurrent_dropout=.2)
    # Add bidirectional
    rnn_one = Bidirectional(rnn_one, name="rnn_bidir_one")(pooled_cnn)
    # Add batch normalization
    bn_rnn_one = BatchNormalization(name='bn_rnn_one')(rnn_one)
    
    rnn_two = GRU(units, activation='relu',
        return_sequences=True, name='rnn_two', dropout=.2, recurrent_dropout=.2)
    # Add bidirectional
    rnn_two = Bidirectional(rnn_two, name="rnn_bidir_two")(bn_rnn_one)
    
    # Add batch normalization
    bn_rnn_two = BatchNormalization(name='bn_rnn_two')(rnn_two)
    
    # Add dense layer (time-distributed)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_two)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, stride=1, pool_size=pool_size, dilation=dilation)
    print(model.summary())
    return model