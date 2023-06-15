import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras import backend as K
from tensorflow.python.ops.gen_math_ops import square
from tensorflow.keras.optimizers import Adam
from keras.layers import Bidirectional
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import AdamW

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
mae, rmse, r2 = mean_absolute_error, mean_squared_error, r2_score 

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Input, TimeDistributed, RepeatVector, Dense
from tensorflow.keras.layers import multiply, concatenate, Flatten, Activation, dot
# ----------------------------------------------------------------------------

def readdata(inputcsv, outputcsv):
    '''
    This function is to read the input_data3.csv and ground_truth3.csv files
    Input: reforecast data csv, reanalysis data csv
    Return: train_dataset df, test_dataset df
    '''

    idx = pd.IndexSlice
    input_data2 = pd.read_csv(inputcsv, index_col=[0], header=[0,1])
    input_data2.index = pd.to_datetime(input_data2.index)
    input_data2.columns = input_data2.columns.set_levels(input_data2.columns.levels[0].astype('int64'), level=0)
    input_data2.columns = input_data2.columns.set_levels(input_data2.columns.levels[1].astype('string'), level=1)

    ground_truth2 = pd.read_csv(outputcsv, index_col=[0], header=[0,1])
    ground_truth2.index = pd.to_datetime(ground_truth2.index)
    ground_truth2.columns = ground_truth2.columns.set_levels(ground_truth2.columns.levels[0].astype('int64'), level=0)
    ground_truth2.columns = ground_truth2.columns.set_levels(ground_truth2.columns.levels[1].astype('string'), level=1)

    log_transform = lambda x: np.log10(x+1) if x.name[1] == 'tp' else x 
    input_data2 = input_data2.apply(log_transform)
    ground_truth2 = ground_truth2.apply(log_transform)

    scaledx = MinMaxScaler()
    scaled_input = scaledx.fit_transform(input_data2.values)
    scaled_input_df = pd.DataFrame(scaled_input, index=input_data2.index, columns=input_data2.columns)

    scaledy = MinMaxScaler()
    scaled_ground = scaledy.fit_transform(ground_truth2.values)
    scaled_ground_df = pd.DataFrame(scaled_ground, index=ground_truth2.index, columns=ground_truth2.columns)

    frames = [scaled_input_df, scaled_ground_df]
    dataset = pd.concat(frames, axis=1)

    train_dataset = dataset.loc['2000-01-01':'2016-12-31'] 
    test_dataset = dataset.loc['2017-01-01':'2019-12-31']

    return train_dataset, test_dataset, scaledx, scaledy
# ----------------------------------------------------------------------------

date_index = pd.date_range('2000-01-01','2016-12-22',freq='D') # changed to 2000-01-01 from 2000-01-10
break_index = [0]+[list(date_index).index(pd.to_datetime('%s-12-31'%year))+1 
 for year in range(2000,2016)] + [len(date_index)]

def split_by_year(freq_year, break_index):
    # freq_year = 2
    end_index = 0
    tscv = []
    while True:
        start_index = end_index
        if start_index + freq_year >= len(break_index):
            break
        end_index = min(start_index+2*freq_year, len(break_index)-1)
        tscv.append((list(range(break_index[start_index],break_index[start_index+freq_year])),
                     list(range(break_index[start_index+freq_year],break_index[end_index]))))
    return tscv
# ----------------------------------------------------------------------------

def get_xy(series, time_step, n_feature):
    x = series.iloc[:,:-1].T.unstack(level=0).T.values.reshape(len(series),time_step,n_feature) # time_step will be 10
    y = pd.concat([series.iloc[:,-1].shift(-i) for i in range(time_step)], axis=1).dropna(axis=0, how='any').values
    y = y.reshape(y.shape[0],y.shape[1],1)
    x = x[:y.shape[0],:,:]
    return x, y

# ----------------------------------------------------------------------------
# weighted MSE loss functions
def my_MSE_weighted2(y_true,y_pred):
    '''
    weighted MSE with 2.0 exp
    '''
    return K.mean(tf.multiply(tf.exp(tf.multiply(2.0, y_true)), tf.square(tf.subtract(y_pred, y_true))))

def my_MSE_weighted1_5(y_true,y_pred):
    '''
    weighted MSE with 1.5 exp
    '''
    return K.mean(tf.multiply(tf.exp(tf.multiply(1.5, y_true)), tf.square(tf.subtract(y_pred, y_true))))
# ----------------------------------------------------------------------------
# custom eval func using RMSE
def my_custom_eval_func(y_true, y_pred):
    '''
    This function (for hyperparameter tuning) 
    evaluates the model based on the returned error metric.
    In this case, we are evaluating based on RMSE
    '''
    # Remove 3D array warning
    if len(y_pred.shape) == 3:
        y_pred = y_pred.reshape(y_pred.shape[:-1])
    return rmse(y_true, y_pred, squared=False)

myenvEstimator  = make_scorer(my_custom_eval_func, greater_is_better=False)

#-----------------------------------------------------------------------------
# THE CODE BELOW IS FOR MEAN SCORE AND AVG STD FOR HYPERPARAMETER TUNING
def mean_score(tparams, tscores, tstd):
  '''
  tparams: unique params combination
  tscores: scores from multiple runs
  tstd: std from multiple runs
  '''
  final_score = [0] * len(tscores[0])
  final_std = [0] * len(tscores[0])
  for i in range(len(tscores)):
    final_score += tscores[i]
    final_std += (tstd[i])**2 # to get variance to find avg std
  return tparams[0], final_score/len(tscores), (final_std/len(tscores))**(1/2)

# ----------------------------------------------------------------------------
# THE CODE BELOW IS FOR TESTING 

# attention
def as2smodel(neurons, activation, loss, optimizer, input, output):
    model = Sequential() 
    encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(neurons, activation=activation, return_state=True, return_sequences=True)(input)
    decoder_input = RepeatVector(output.shape[1])(encoder_last_h)
    decoder_stack_h = LSTM(neurons, activation=activation, return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
    attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
    attention = Activation('softmax')(attention)
    context = dot([attention, encoder_stack_h], axes=[2,1])
    decoder_combined_context = concatenate([context, decoder_stack_h])

    out = TimeDistributed(Dense(output.shape[2]))(decoder_combined_context)
    model = Model(inputs=input, outputs=out)
    model.compile(loss=loss, optimizer=optimizer) 
    return model

def bias2smodel(neurons, activation, loss, opt, input, output):
    model = Sequential() 
    # forward_h, forward_c, backward_h, backward_c
    encoder_stack_h, encoder_forward_h, encoder_forward_c, encoder_backward_h, encoder_backward_c = Bidirectional(LSTM(neurons, activation=activation, return_state=True, return_sequences=True))(input)
    encoder_last_h = concatenate([encoder_forward_h, encoder_backward_h]) 
    encoder_last_c = concatenate([encoder_forward_c, encoder_backward_c])

    decoder_input = RepeatVector(output.shape[1])(encoder_last_h)
    decoder_stack_h = LSTM(neurons*2, activation=activation, return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
    attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
    attention = Activation('softmax')(attention)
    context = dot([attention, encoder_stack_h], axes=[2,1])
    decoder_combined_context = concatenate([context, decoder_stack_h])

    out = TimeDistributed(Dense(output.shape[2]))(decoder_combined_context)
    model = Model(inputs=input, outputs=out)
    model.compile(loss=loss, optimizer=opt) # adamW mse2
    return model

def yhat_y_tp(fmodel, scalex, scaley, testx, testy):
    '''
    This function returns the original values for precipitation 
    Return:
    original_test_x --> original 5 variables values
    original_test_yhat --> corrected tp value
    original_test_y --> reanalysis (ground truth) truth tp value
    original_test_x_tp --> reforecast tp variable value
    '''
    pred_e1d1 = fmodel.predict(testx)

    # forecast (leadtime1) vs groundtruth
    # t2m	tp	H	C	E
    original_test_x = scalex.inverse_transform(testx[:,:,[3,4,2,0,1]].reshape(testx.shape[0], testx.shape[1]*testx.shape[2]))
    original_test_x[:, [1,6,11,16,21,26,31,36,41,46]] = 10**original_test_x[:, [1,6,11,16,21,26,31,36,41,46]] -1 # log scaling back to original tp for test_x

    original_test_y = scaley.inverse_transform(testy[:,:,0])
    original_test_y = 10** original_test_y -1 # log scaling back for test_y

    original_test_yhat = scaley.inverse_transform(pred_e1d1[:,:,0])
    original_test_yhat = 10**original_test_yhat - 1 # log scaling for correction
    original_test_yhat[original_test_yhat<=0] = 0

    original_test_x_tp = original_test_x[:, [1,6,11,16,21,26,31,36,41,46]] # creating a DF for the original_test_x with only tp variable
    return pred_e1d1, original_test_x, original_test_yhat, original_test_y, original_test_x_tp