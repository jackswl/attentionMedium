from utils import *
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input, TimeDistributed, RepeatVector, Dense
from tensorflow.keras.layers import multiply, concatenate, Flatten, Activation, dot
from sklearn.model_selection import GridSearchCV

# this py file consists of both tuning and testing of lightgbm model

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

test_x = test_x.reshape(test_x.shape[0], test_x.shape[1]*test_x.shape[2])
test_y = test_y.reshape(test_y.shape[0], test_y.shape[1]*test_y.shape[2])

np.random.seed(42)
# tf.random.set_seed(42)

# --------------------------------------------------------
# tuning
estimator = LGBMRegressor(learning_rate=0.1)
lgb = MultiOutputRegressor(estimator=estimator)

parameters = {'estimator__n_estimators':np.arange(100,1001,100), 'estimator__max_depth':np.arange(1,9,1), 'estimator__min_data_in_leaf':np.arange(1,41,5),
              'estimator__num_leaves':np.arange(2,22,2)} # actual calc

clf = GridSearchCV(lgb, parameters, cv=tscv, scoring='neg_root_mean_squared_error', verbose=0)
clf.fit(train_x, train_y)

# --------------------------------------------------------
# hyperparameter tuning results
results_df = pd.DataFrame(clf.cv_results_)
results_df['mean_test_score'] = results_df['mean_test_score'].abs()
results_df = results_df.sort_values(by=["rank_test_score"])
results_df = results_df.set_index(
    results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
).rename_axis("kernel")
results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]

model_df = results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]
model_df.to_csv('/home/svu/e0560091/hpc_tunings_lgbm_v2.csv')

# --------------------------------------------------------
# results
pred_e1d1 = clf.predict(test_x) 
test_x = test_x.reshape(1086,10,5)
test_y = test_y.reshape(1086,10,1)
pred_e1d1 = pred_e1d1.reshape(1086,10,1)

original_test_x = scaledx.inverse_transform(test_x[:,:,[3,4,2,0,1]].reshape(test_x.shape[0], test_x.shape[1]*test_x.shape[2]))
original_test_x[:, [1,6,11,16,21,26,31,36,41,46]] = 10**original_test_x[:, [1,6,11,16,21,26,31,36,41,46]] -1 # log scaling back to original tp for test_x

original_test_y = scaledy.inverse_transform(test_y[:,:,0])
original_test_y = 10** original_test_y -1 # log scaling back for test_y

original_test_yhat = scaledy.inverse_transform(pred_e1d1[:,:,0])
original_test_yhat = 10**original_test_yhat - 1 # log scaling for correction
original_test_yhat[original_test_yhat<=0] = 0
original_test_x_tp = original_test_x[:, [1,6,11,16,21,26,31,36,41,46]] # creating a DF for the original_test_x with only tp variable
#-----------------------------------------------------------------------------

'''
regression results
'''
# This part output the assessment result TABLE FOR MAE RMSE AND R2
names = locals()
idx = pd.IndexSlice
result = pd.DataFrame(index=range(1,11),columns=
             pd.MultiIndex.from_product([['mae', 'rmse', 'r2'],
                                         ['original', 'correct']]))
 
original_test_x_tp = original_test_x[:, [1,6,11,16,21,26,31,36,41,46]] # creating a DF for the original_test_x with only tp variable


for index in ['mae', 'rmse', 'r2']:
    if index != 'rmse':
        result.loc[:,idx[index, 'original']] = [names[index](original_test_y[:,i], original_test_x_tp[:,i])  # 1 --> i
                                                for i in range(time_step)]
        result.loc[:,idx[index, 'correct']] = [names[index](original_test_y[:,i], original_test_yhat[:,i]) 
                                               for i in range(time_step)]
    else:
        result.loc[:,idx[index, 'original']] = [names[index](original_test_y[:,i], original_test_x_tp[:,i], # 1 --> i
                                                            squared=False) 
                                            for i in range(time_step)]
        result.loc[:,idx[index, 'correct']] = [names[index](original_test_y[:,i], original_test_yhat[:,i],
                                                           squared=False) 
                                           for i in range(time_step)]
result = result.round(3)
result.to_excel('/home/svu/e0560091/hpc_re_lgbm_v2.xlsx') # re = regression
#-----------------------------------------------------------------------------

'''
classification results
'''
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# corrected tp (predicted tp)
original_test_yhat = scaledy.inverse_transform(pred_e1d1[:,:,0])
original_test_yhat = 10**original_test_yhat - 1 # log scaling correction
original_test_yhat[original_test_yhat<=0] = 0
corrected_df = pd.DataFrame(original_test_yhat)
corrected_df = pd.DataFrame(pd.cut(corrected_df[i], bins=[-1,0,7.5,50,1000],labels=['A','B','C','D']) for i in range(10)).T 
# corrected_df = pd.DataFrame(pd.cut(corrected_df[i], bins=[-1,5,20,50,1000],labels=['A','B','C','D']) for i in range(10)).T # indonesia paper

# # reforecast tp
reforecasted_df = pd.DataFrame(original_test_x[:, 1::5]) 
reforecasted_df = pd.DataFrame(pd.cut(reforecasted_df[i], bins=[-1,0,7.5,50,1000],labels=['A','B','C','D']) for i in range(10)).T
# reforecasted_df = pd.DataFrame(pd.cut(reforecasted_df[i], bins=[-1,5,20,50,1000],labels=['A','B','C','D']) for i in range(10)).T

# # reforecasted_df.loc[reforecasted_df[0] > 64.4] # to check values

# # ground truth test set
original_test_y = scaledy.inverse_transform(test_y[:,:,0])
original_test_y = 10** original_test_y -1 # log scaling back for test_y
real_df = pd.DataFrame(original_test_y)
real_df = pd.DataFrame(pd.cut(real_df[i], bins=[-1,0,7.5,50,1000],labels=['A','B','C','D']) for i in range(10)).T
# real_df = pd.DataFrame(pd.cut(real_df[i], bins=[-1,5,20,50,1000],labels=['A','B','C','D']) for i in range(10)).T

# !pip install xlsxwriter
import warnings
warnings.filterwarnings('ignore')
def get_classification_metrics(confusion_matrix):
    # the number of false positives
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix) # axis=0 row
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    return TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC

metrics  = ['TPR', 'TNR', 'PPV', 'NPV', 'FPR', 'FNR', 'FDR', 'ACC']
labels = ['No rain', 'Light', 'Moderate','Heavy']
types = ['original', 'correct']
classification_metrics = pd.DataFrame(index=range(10),
                                      columns = pd.MultiIndex.from_product([metrics, types, labels]))
for i in range(time_step):
    cm = confusion_matrix(y_true=real_df.loc[:,i], y_pred=reforecasted_df.loc[:,i])
    cm_corrected = confusion_matrix(y_true=real_df.loc[:,i], y_pred=corrected_df.loc[:,i])
    TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC = get_classification_metrics(cm)
    for m in metrics:
        classification_metrics.loc[i, idx[m, 'original', :]] = names[m]
    TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC = get_classification_metrics(cm_corrected)
    for m in metrics:
        classification_metrics.loc[i, idx[m, 'correct', :]] = names[m]

# classification_metrics.astype('float').round(3).loc[:,idx['FPR',:,:]].to_excel('Classification.xlsx')
df1 = classification_metrics.astype('float').round(3).loc[:,idx['FPR',:,:]]
df2 = classification_metrics.astype('float').round(3).loc[:,idx['ACC',:,:]]
df3 = classification_metrics.astype('float').round(3).loc[:,idx['TPR',:,:]]
df4 = classification_metrics.astype('float').round(3).loc[:,idx['TNR',:,:]]
df5 = classification_metrics.astype('float').round(3).loc[:,idx['PPV',:,:]]
df6 = classification_metrics.astype('float').round(3).loc[:,idx['FDR',:,:]]
df7 = classification_metrics.astype('float').round(3).loc[:,idx['NPV',:,:]]
df8 = classification_metrics.astype('float').round(3).loc[:,idx['FNR',:,:]]

# exporting to excel with different sheets (metrics)
writer = pd.ExcelWriter('/home/svu/e0560091/hpc_cl_lgbm_v2.xlsx', engine = 'xlsxwriter') # cl = classification
df1.to_excel(writer, sheet_name = 'FPR')
df2.to_excel(writer, sheet_name = 'ACC')
df3.to_excel(writer, sheet_name = 'TPR')
df4.to_excel(writer, sheet_name = 'TNR')
df5.to_excel(writer, sheet_name = 'PPV')
df6.to_excel(writer, sheet_name = 'FDR')
df7.to_excel(writer, sheet_name = 'NPV')
df8.to_excel(writer, sheet_name = 'FNR')

writer.save()
writer.close()
#-----------------------------------------------------------------------------