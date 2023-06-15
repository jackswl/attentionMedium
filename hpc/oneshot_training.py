from utils import *

from sklearn.model_selection import GridSearchCV
'''

This py file (oneshot_training.py) contains all the training files merged into 1
As such, both Adam and AdamW will be tuned in this py file. 

4 Outputs:
'/home/svu/e0560091/hpc_tuning_adamw_v2.csv'
'/home/svu/e0560091/hpc_tuning_adam_v2.csv'
'/home/svu/e0560091/hpc_tuningbi_adamw_v2.csv'
'/home/svu/e0560091/hpc_tuningbi_adam_v2.csv'

'''
#-----------------------------------------------------------------------------
# FIRST ITERATION 
# input 
train_dataset, test_dataset, scaledx, scaledy = readdata('/home/svu/e0560091/input_data3.csv', '/home/svu/e0560091/ground_truth3.csv')

# tscv split
freq_year = 2 # freq_year < total year/2  # 2000-2001 (2 years) test 2 years x4
tscv = split_by_year(freq_year, break_index)

time_step, n_features = 10, 5
train_x, train_y = get_xy(train_dataset, time_step, n_features)
test_x, test_y = get_xy(test_dataset, time_step, n_features)

input_train = Input(shape=(train_x.shape[1], train_x.shape[2]))
output_train = Input(shape=(train_y.shape[1], train_y.shape[2]))

# gridsearchCV does not take in 3D shape as inputs, only 2D
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
train_y = train_y.reshape(train_y.shape[0], train_y.shape[1]*train_y.shape[2])

from sklearn.base import BaseEstimator, ClassifierMixin
class mylstm(BaseEstimator, ClassifierMixin):
    '''
    hyperparameter tuning tscv
    '''
    def __init__(self, n_steps=10, n_features=5, 
                 activation='relu', optimizer='adam',loss=my_MSE_weighted2,
                 lstm=32, dense=1, verbose=1,
                 epochs=20, batch_size=8):
                 #learning_rate=1e-3, #weight_decay=1e-5):
        
        # static parameters
        self.n_steps = n_steps
        self.n_features = n_features
        self.verbose = verbose
        
        # Parameters that can be optimized
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.lstm = lstm
        self.dense = dense
        self.epochs = epochs
        self.batch_size = batch_size
  
        #---------- luong attention ----------------
        encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(self.lstm, activation=self.activation, return_state=True, return_sequences=True)(input_train)
        decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
        decoder_stack_h = LSTM(self.lstm, activation=self.activation, return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, encoder_stack_h], axes=[2,1])

        decoder_combined_context = concatenate([context, decoder_stack_h])
        out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)
        self.model = Model(inputs=input_train, outputs=out)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, X, y, **kw):
        X = X.reshape(X.shape[0], self.n_steps, self.n_features)
        
        # Control display output, If parameter `verbose` is given, it will be used. 
        # If no `verbose` is given, the default value of the class is used.
        if 'verbose' in kw.keys():
            return self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, **kw)
        else:
            return self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, **kw)
        

    def predict(self, X, **kw):
        X = X.reshape(X.shape[0], self.n_steps, self.n_features)
        
        if 'verbose' in kw.keys():
            return self.model.predict(X, **kw)
        else:
            return self.model.predict(X, verbose=self.verbose, **kw)
    
    def score(self, X, y, **kw):
        X = X.reshape(X.shape[0], self.n_steps, self.n_features)
        
        # Control display output
        if 'verbose' in kw.keys():
            return self.model.evaluate(X, y, **kw)
        else:
            return self.model.evaluate(X, y, verbose=self.verbose, **kw)
#-----------------------------------------------------------------------------

lr_schedule = tf.optimizers.schedules.ExponentialDecay(1e-3, 24, 0.95) # every 2 epochs
wd_schedule = tf.optimizers.schedules.ExponentialDecay(1e-5, 24, 0.95)
optW = AdamW(learning_rate=lr_schedule, weight_decay=lambda : None)
optW.weight_decay = lambda : wd_schedule(optW.iterations)
# opt = Adam(learning_rate=1e-4)

# parameters = {'lstm':[48,64,80,100,128,144,160]} # Actual calculation
parameters = {'activation':('relu','tanh'), 'lstm':[32,48,64,80,96,112,128,144], 'loss':(my_MSE_weighted2, 'mse')} # Actual calculation
# parameters = {'activation':('relu','tanh'), 'lstm':[32,48]} # Actual calculation

NUM_TRIALS = 30
scores = []
params = []
std = []
for i in range(NUM_TRIALS):
     tscv = split_by_year(freq_year, break_index)
     clf = GridSearchCV(mylstm(verbose=1, epochs=20, batch_size=8, optimizer=optW), parameters, cv=tscv, scoring=myenvEstimator)
     clf.fit(train_x, train_y)
     scores.append(clf.cv_results_['mean_test_score'])
     params.append(clf.cv_results_['params'])
     std.append(clf.cv_results_['std_test_score'])
#-----------------------------------------------------------------------------
# hyperparameter tuning results

a, b, c = mean_score(params, scores, std)

results_df = pd.DataFrame(
            {'params': a,
             'avg_scores': b,
             'avg_std': c}).sort_values("avg_scores", ascending=False)

results_df.to_csv('/home/svu/e0560091/1_tuning_adamw.csv')

#-------------------------------------------------------------------------------
# SECOND ITERATION
# COMMENTED FROM HERE 14/8
train_dataset, test_dataset, scaledx, scaledy = readdata('/home/svu/e0560091/input_data3.csv', '/home/svu/e0560091/ground_truth3.csv')

# tscv split
freq_year = 2 # freq_year < total year/2  # 2000-2001 (2 years) test 2 years x4
tscv = split_by_year(freq_year, break_index)

time_step, n_features = 10, 5
train_x, train_y = get_xy(train_dataset, time_step, n_features)
test_x, test_y = get_xy(test_dataset, time_step, n_features)

input_train = Input(shape=(train_x.shape[1], train_x.shape[2]))
output_train = Input(shape=(train_y.shape[1], train_y.shape[2]))

# gridsearchCV does not take in 3D shape as inputs, only 2D
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
train_y = train_y.reshape(train_y.shape[0], train_y.shape[1]*train_y.shape[2])


from sklearn.base import BaseEstimator, ClassifierMixin
class mylstm(BaseEstimator, ClassifierMixin):
    '''
    hyperparameter tuning tscv
    '''
    def __init__(self, n_steps=10, n_features=5, 
                 activation='relu', optimizer='adam',loss=my_MSE_weighted2,
                 lstm=32, dense=1, verbose=1,
                 epochs=20, batch_size=8):
                 #learning_rate=1e-3, #weight_decay=1e-5):
        
        # static parameters
        self.n_steps = n_steps
        self.n_features = n_features
        self.verbose = verbose
        
        # Parameters that can be optimized
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.lstm = lstm
        self.dense = dense
        self.epochs = epochs
        self.batch_size = batch_size
  
        #---------- luong attention ----------------
        encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(self.lstm, activation=self.activation, return_state=True, return_sequences=True)(input_train)
        decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
        decoder_stack_h = LSTM(self.lstm, activation=self.activation, return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, encoder_stack_h], axes=[2,1])

        decoder_combined_context = concatenate([context, decoder_stack_h])
        out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)
        self.model = Model(inputs=input_train, outputs=out)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, X, y, **kw):
        X = X.reshape(X.shape[0], self.n_steps, self.n_features)
        
        # Control display output, If parameter `verbose` is given, it will be used. 
        # If no `verbose` is given, the default value of the class is used.
        if 'verbose' in kw.keys():
            return self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, **kw)
        else:
            return self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, **kw)
        

    def predict(self, X, **kw):
        X = X.reshape(X.shape[0], self.n_steps, self.n_features)
        
        if 'verbose' in kw.keys():
            return self.model.predict(X, **kw)
        else:
            return self.model.predict(X, verbose=self.verbose, **kw)
    
    def score(self, X, y, **kw):
        X = X.reshape(X.shape[0], self.n_steps, self.n_features)
        
        # Control display output
        if 'verbose' in kw.keys():
            return self.model.evaluate(X, y, **kw)
        else:
            return self.model.evaluate(X, y, verbose=self.verbose, **kw)
#-----------------------------------------------------------------------------
opt = Adam(learning_rate=1e-4)

# parameters = {'lstm':[48,64,80,100,128,144,160]} # Actual calculation
parameters = {'activation':('relu','tanh'), 'lstm':[32,48,64,80,96,112,128,144], 'loss':(my_MSE_weighted2, 'mse')} # Actual calculation
# parameters = {'activation':('relu','tanh'), 'lstm':[32,48], 'loss':(my_MSE_weighted2, 'mse')} # Actual calculation

NUM_TRIALS = 30
scores = []
params = []
std = []
for i in range(NUM_TRIALS):
     tscv = split_by_year(freq_year, break_index)
     clf = GridSearchCV(mylstm(verbose=1, epochs=20, batch_size=8, optimizer=opt), parameters, cv=tscv, scoring=myenvEstimator)
     clf.fit(train_x, train_y)
     scores.append(clf.cv_results_['mean_test_score'])
     params.append(clf.cv_results_['params'])
     std.append(clf.cv_results_['std_test_score'])
#-----------------------------------------------------------------------------
# hyperparameter tuning results

a, b, c = mean_score(params, scores, std)

results_df = pd.DataFrame(
            {'params': a,
             'avg_scores': b,
             'avg_std': c}).sort_values("avg_scores", ascending=False)

results_df.to_csv('/home/svu/e0560091/2_tuning_adam.csv')
#-----------------------------------------------------------------------------
# THIRD ITERATION (bi-A2S for AdamW)

# input 
train_dataset, test_dataset, scaledx, scaledy = readdata('/home/svu/e0560091/input_data3.csv', '/home/svu/e0560091/ground_truth3.csv')

# tscv split
freq_year = 2 # freq_year < total year/2  # 2000-2001 (2 years) test 2 years x4
tscv = split_by_year(freq_year, break_index)

time_step, n_features = 10, 5
train_x, train_y = get_xy(train_dataset, time_step, n_features)
test_x, test_y = get_xy(test_dataset, time_step, n_features)

input_train = Input(shape=(train_x.shape[1], train_x.shape[2]))
output_train = Input(shape=(train_y.shape[1], train_y.shape[2]))

# gridsearchCV does not take in 3D shape as inputs, only 2D
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
train_y = train_y.reshape(train_y.shape[0], train_y.shape[1]*train_y.shape[2])


from sklearn.base import BaseEstimator, ClassifierMixin
class mylstm(BaseEstimator, ClassifierMixin):
    '''
    hyperparameter tuning tscv
    '''
    def __init__(self, n_steps=10, n_features=5, 
                 activation='relu', optimizer='adam',loss=my_MSE_weighted2,
                 lstm=32, dense=1, verbose=1,
                 epochs=20, batch_size=8):
                 #learning_rate=1e-3, #weight_decay=1e-5):
        
        # static parameters
        self.n_steps = n_steps
        self.n_features = n_features
        self.verbose = verbose
        
        # Parameters that can be optimized
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.lstm = lstm
        self.dense = dense
        self.epochs = epochs
        self.batch_size = batch_size
  
        #---------- luong attention ----------------
        encoder_stack_h, encoder_forward_h, encoder_forward_c, encoder_backward_h, encoder_backward_c = Bidirectional(LSTM(self.lstm, activation=self.activation, 
                                                                                                                           return_state=True, return_sequences=True))(input_train)
        encoder_last_h = concatenate([encoder_forward_h, encoder_backward_h]) 
        encoder_last_c = concatenate([encoder_forward_c, encoder_backward_c])

        decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
        decoder_stack_h = LSTM(self.lstm*2, activation=self.activation, return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, encoder_stack_h], axes=[2,1])
        decoder_combined_context = concatenate([context, decoder_stack_h])
        out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)
        self.model = Model(inputs=input_train, outputs=out)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)    

    def fit(self, X, y, **kw):
        X = X.reshape(X.shape[0], self.n_steps, self.n_features)
        
        # Control display output, If parameter `verbose` is given, it will be used. 
        # If no `verbose` is given, the default value of the class is used.
        if 'verbose' in kw.keys():
            return self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, **kw)
        else:
            return self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, **kw)
        

    def predict(self, X, **kw):
        X = X.reshape(X.shape[0], self.n_steps, self.n_features)
        
        if 'verbose' in kw.keys():
            return self.model.predict(X, **kw)
        else:
            return self.model.predict(X, verbose=self.verbose, **kw)
    
    def score(self, X, y, **kw):
        X = X.reshape(X.shape[0], self.n_steps, self.n_features)
        
        # Control display output
        if 'verbose' in kw.keys():
            return self.model.evaluate(X, y, **kw)
        else:
            return self.model.evaluate(X, y, verbose=self.verbose, **kw)
#-----------------------------------------------------------------------------

lr_schedule = tf.optimizers.schedules.ExponentialDecay(1e-3, 24, 0.95) # every 2 epochs fixed
wd_schedule = tf.optimizers.schedules.ExponentialDecay(1e-5, 24, 0.95)
optW = AdamW(learning_rate=lr_schedule, weight_decay=lambda : None)
optW.weight_decay = lambda : wd_schedule(optW.iterations)
# opt = Adam(learning_rate=1e-4)

# parameters = {'lstm':[48,64,80,100,128,144,160]} # Actual calculation
parameters = {'activation':('relu','tanh'), 'lstm':[32,48,64,80,96,112,128,144], 'loss':(my_MSE_weighted2, 'mse')} # Actual calculation
# parameters = {'activation':('relu','tanh'), 'lstm':[32,48], 'loss':(my_MSE_weighted2, 'mse')} # Actual calculation

NUM_TRIALS = 30
scores = []
params = []
std = []
for i in range(NUM_TRIALS):
     tscv = split_by_year(freq_year, break_index)
     clf = GridSearchCV(mylstm(verbose=1, epochs=20, batch_size=8, optimizer=optW), parameters, cv=tscv, scoring=myenvEstimator)
     clf.fit(train_x, train_y)
     scores.append(clf.cv_results_['mean_test_score'])
     params.append(clf.cv_results_['params'])
     std.append(clf.cv_results_['std_test_score'])
#-----------------------------------------------------------------------------
# hyperparameter tuning results

a, b, c = mean_score(params, scores, std)

results_df = pd.DataFrame(
            {'params': a,
             'avg_scores': b,
             'avg_std': c}).sort_values("avg_scores", ascending=False)

results_df.to_csv('/home/svu/e0560091/3_tuningbi_adamw.csv')

#-----------------------------------------------------------------------------
# FOURTH ITERATION (bi_A2S Adam)

# input 
train_dataset, test_dataset, scaledx, scaledy = readdata('/home/svu/e0560091/input_data3.csv', '/home/svu/e0560091/ground_truth3.csv')

# tscv split
freq_year = 2 # freq_year < total year/2  # 2000-2001 (2 years) test 2 years x4
tscv = split_by_year(freq_year, break_index)

time_step, n_features = 10, 5
train_x, train_y = get_xy(train_dataset, time_step, n_features)
test_x, test_y = get_xy(test_dataset, time_step, n_features)

input_train = Input(shape=(train_x.shape[1], train_x.shape[2]))
output_train = Input(shape=(train_y.shape[1], train_y.shape[2]))

# gridsearchCV does not take in 3D shape as inputs, only 2D
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
train_y = train_y.reshape(train_y.shape[0], train_y.shape[1]*train_y.shape[2])


from sklearn.base import BaseEstimator, ClassifierMixin
class mylstm(BaseEstimator, ClassifierMixin):
    '''
    hyperparameter tuning tscv
    '''
    def __init__(self, n_steps=10, n_features=5, 
                 activation='relu', optimizer='adam',loss=my_MSE_weighted2,
                 lstm=32, dense=1, verbose=1,
                 epochs=20, batch_size=8):
                 #learning_rate=1e-3, #weight_decay=1e-5):
        
        # static parameters
        self.n_steps = n_steps
        self.n_features = n_features
        self.verbose = verbose
        
        # Parameters that can be optimized
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.lstm = lstm
        self.dense = dense
        self.epochs = epochs
        self.batch_size = batch_size
  
        #---------- luong attention ----------------
        encoder_stack_h, encoder_forward_h, encoder_forward_c, encoder_backward_h, encoder_backward_c = Bidirectional(LSTM(self.lstm, activation=self.activation, 
                                                                                                                           return_state=True, return_sequences=True))(input_train)
        encoder_last_h = concatenate([encoder_forward_h, encoder_backward_h]) 
        encoder_last_c = concatenate([encoder_forward_c, encoder_backward_c])

        decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
        decoder_stack_h = LSTM(self.lstm*2, activation=self.activation, return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, encoder_stack_h], axes=[2,1])
        decoder_combined_context = concatenate([context, decoder_stack_h])
        out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)
        self.model = Model(inputs=input_train, outputs=out)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)    

    def fit(self, X, y, **kw):
        X = X.reshape(X.shape[0], self.n_steps, self.n_features)
        
        # Control display output, If parameter `verbose` is given, it will be used. 
        # If no `verbose` is given, the default value of the class is used.
        if 'verbose' in kw.keys():
            return self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, **kw)
        else:
            return self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, **kw)
        

    def predict(self, X, **kw):
        X = X.reshape(X.shape[0], self.n_steps, self.n_features)
        
        if 'verbose' in kw.keys():
            return self.model.predict(X, **kw)
        else:
            return self.model.predict(X, verbose=self.verbose, **kw)
    
    def score(self, X, y, **kw):
        X = X.reshape(X.shape[0], self.n_steps, self.n_features)
        
        # Control display output
        if 'verbose' in kw.keys():
            return self.model.evaluate(X, y, **kw)
        else:
            return self.model.evaluate(X, y, verbose=self.verbose, **kw)
#-----------------------------------------------------------------------------

# lr_schedule = tf.optimizers.schedules.ExponentialDecay(1e-3, 24, 0.95) # every 2 epochs fixed
# wd_schedule = tf.optimizers.schedules.ExponentialDecay(1e-5, 24, 0.95)
# optW = AdamW(learning_rate=lr_schedule, weight_decay=lambda : None)
# optW.weight_decay = lambda : wd_schedule(optW.iterations)
opt = Adam(learning_rate=1e-4)

# parameters = {'lstm':[48,64,80,100,128,144,160]} # Actual calculation
parameters = {'activation':('relu','tanh'), 'lstm':[32,48,64,80,96,112,128,144], 'loss':(my_MSE_weighted2, 'mse')} # Actual calculation
# parameters = {'activation':('relu','tanh'), 'lstm':[32,48], 'loss':(my_MSE_weighted2, 'mse')} # Actual calculation

NUM_TRIALS = 30
scores = []
params = []
std = []
for i in range(NUM_TRIALS):
     tscv = split_by_year(freq_year, break_index)
     clf = GridSearchCV(mylstm(verbose=1, epochs=20, batch_size=8, optimizer=opt), parameters, cv=tscv, scoring=myenvEstimator)
     clf.fit(train_x, train_y)
     scores.append(clf.cv_results_['mean_test_score'])
     params.append(clf.cv_results_['params'])
     std.append(clf.cv_results_['std_test_score'])
#-----------------------------------------------------------------------------
# hyperparameter tuning results

a, b, c = mean_score(params, scores, std)

results_df = pd.DataFrame(
            {'params': a,
             'avg_scores': b,
             'avg_std': c}).sort_values("avg_scores", ascending=False)

results_df.to_csv('/home/svu/e0560091/4_tuningbi_adam.csv')