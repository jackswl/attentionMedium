from utils import *

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

'''
4 testing runs as per oneshot_training.py
IN THIS ORDER:
1st run --> '/home/svu/e0560091/hpc_tuning_adamw_v2.csv'
2nd run --> '/home/svu/e0560091/hpc_tuning_adam_v2.csv'
3rd run --> '/home/svu/e0560091/hpc_tuningbi_adamw_v2.csv'
4th run --> '/home/svu/e0560091/hpc_tuningbi_adam_v2.csv'
'''
#-----------------------------------------------------------------------------

train_dataset, test_dataset, scaledx, scaledy = readdata('/home/svu/e0560091/input_data3.csv', '/home/svu/e0560091/ground_truth3.csv')

time_step, n_features = 10, 5
train_x, train_y = get_xy(train_dataset, time_step, n_features)
test_x, test_y = get_xy(test_dataset, time_step, n_features)

input_train = Input(shape=(train_x.shape[1], train_x.shape[2]))
output_train = Input(shape=(train_y.shape[1], train_y.shape[2]))

np.random.seed(42)
tf.random.set_seed(42)

# Adam optimizer
opt = Adam(learning_rate=1e-4)
#-----------------------------------------------------------------------------

# attention model
model = bias2smodel(128, 'tanh', my_MSE_weighted2, opt, input_train, output_train)
model.fit(train_x, train_y, epochs=160, batch_size=64, verbose=1)
#-----------------------------------------------------------------------------

# original y, original yhat and original tp
pred_e1d1, original_test_x, original_test_yhat, original_test_y, original_test_x_tp = yhat_y_tp(model, scaledx, scaledy, test_x, test_y)

#------------------------------------------------------------- plot ----------------------------------------------------------------
fig, ax = plt.subplots(5,2,figsize=(8,16))

x = 0
for i in range(5):
  for j in range(2):
    ax[i,j].set_xlim(0,60)
    ax[i,j].set_ylim(0,60)
    ax[i,j].set_title('Leadtime' + str(x+1))

    ax[i,1].get_yaxis().set_visible(False)
    ax[i,j].get_xaxis().set_visible(False)
#     ############################################################################
    ax[i,j].plot(ax[i,j].get_xlim(), ax[i,j].get_ylim(), ls="--", c=".3")
    ax[i,j].scatter(original_test_x_tp[:,x], original_test_y[:,x], alpha=0.4, label='GEFS reforecast',s=20,color='grey') 
    ax[i,j].scatter(original_test_yhat[:,x], original_test_y[:,x], alpha=0.1, label='Corrected reforecast',s=20, color='blue') # changed to idx0 for test_yhat
    ax[i,j].text(2, 54, r'Reforecast $R^2$ = '+ str(round(r2(original_test_y[:,x], original_test_x_tp[:,x]),3)), fontsize=10)
    ax[i,j].text(2, 50, r'Corrected $R^2$ = '+ str(round(r2(original_test_y[:,x], original_test_yhat[:,x]),3)), fontsize=10)
    x += 1

ax[4,0].get_xaxis().set_visible(True)
ax[4,1].get_xaxis().set_visible(True)
ax[4,0].set_xlabel('Corrected', fontsize=12, labelpad=14)
ax[2,0].set_ylabel('Reforecast', fontsize=12, labelpad=14)
# ax[0,0].text(2, 54, r'Reforecast $R^2$ = '+ str(round(r2(original_test_y[:,0], original_test_x_tp[:,0]),3)), fontsize=10)
# ax[0,0].text(2, 50, r'Corrected $R^2$ = '+ str(round(r2(original_test_y[:,0], original_test_yhat[:,0]),3)), fontsize=10)
fig.tight_layout()
# fig.savefig("testimage.png") #save as jpg
fig.savefig('biadamtanh_mse2.png',dpi=600,bbox_inches = 'tight')

# ------------------------ This part output the assessment result TABLE FOR MAE, RMSE, AND R2 ---------------------------------------
names = locals()
idx = pd.IndexSlice
result = pd.DataFrame(index=range(1,11),columns=
             pd.MultiIndex.from_product([['mae', 'rmse', 'r2'],
                                         ['original', 'correct']]))


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
result.to_excel('/home/svu/e0560091/testingtanh_re_biadam.xlsx')


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
original_test_y = 10**original_test_y -1 # log scaling back for test_y
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
writer = pd.ExcelWriter('/home/svu/e0560091/testingtanh_cl_biadam.xlsx', engine = 'xlsxwriter')
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


# ---------- 2nd iteration ----------------

train_dataset, test_dataset, scaledx, scaledy = readdata('/home/svu/e0560091/input_data3.csv', '/home/svu/e0560091/ground_truth3.csv')

time_step, n_features = 10, 5
train_x, train_y = get_xy(train_dataset, time_step, n_features)
test_x, test_y = get_xy(test_dataset, time_step, n_features)

input_train = Input(shape=(train_x.shape[1], train_x.shape[2]))
output_train = Input(shape=(train_y.shape[1], train_y.shape[2]))

np.random.seed(42)
tf.random.set_seed(42)

# Adam optimizer
opt = Adam(learning_rate=1e-4)
#-----------------------------------------------------------------------------

# attention model
model = bias2smodel(128, 'relu', my_MSE_weighted2, opt, input_train, output_train)
model.fit(train_x, train_y, epochs=160, batch_size=64, verbose=1)
#-----------------------------------------------------------------------------

# original y, original yhat and original tp
pred_e1d1, original_test_x, original_test_yhat, original_test_y, original_test_x_tp = yhat_y_tp(model, scaledx, scaledy, test_x, test_y)

#------------------------------------------------------------- plot ----------------------------------------------------------------
fig, ax = plt.subplots(5,2,figsize=(8,16))

x = 0
for i in range(5):
  for j in range(2):
    ax[i,j].set_xlim(0,60)
    ax[i,j].set_ylim(0,60)
    ax[i,j].set_title('Leadtime' + str(x+1))

    ax[i,1].get_yaxis().set_visible(False)
    ax[i,j].get_xaxis().set_visible(False)
#     ############################################################################
    ax[i,j].plot(ax[i,j].get_xlim(), ax[i,j].get_ylim(), ls="--", c=".3")
    ax[i,j].scatter(original_test_x_tp[:,x], original_test_y[:,x], alpha=0.4, label='GEFS reforecast',s=20,color='grey') 
    ax[i,j].scatter(original_test_yhat[:,x], original_test_y[:,x], alpha=0.1, label='Corrected reforecast',s=20, color='blue') # changed to idx0 for test_yhat
    ax[i,j].text(2, 54, r'Reforecast $R^2$ = '+ str(round(r2(original_test_y[:,x], original_test_x_tp[:,x]),3)), fontsize=10)
    ax[i,j].text(2, 50, r'Corrected $R^2$ = '+ str(round(r2(original_test_y[:,x], original_test_yhat[:,x]),3)), fontsize=10)
    x += 1

ax[4,0].get_xaxis().set_visible(True)
ax[4,1].get_xaxis().set_visible(True)
ax[4,0].set_xlabel('Corrected', fontsize=12, labelpad=14)
ax[2,0].set_ylabel('Reforecast', fontsize=12, labelpad=14)
# ax[0,0].text(2, 54, r'Reforecast $R^2$ = '+ str(round(r2(original_test_y[:,0], original_test_x_tp[:,0]),3)), fontsize=10)
# ax[0,0].text(2, 50, r'Corrected $R^2$ = '+ str(round(r2(original_test_y[:,0], original_test_yhat[:,0]),3)), fontsize=10)
fig.tight_layout()
# fig.savefig("testimage.png") #save as jpg
fig.savefig('biadamrelu_mse2.png',dpi=600,bbox_inches = 'tight')

# ------------------------ This part output the assessment result TABLE FOR MAE, RMSE, AND R2 ---------------------------------------
names = locals()
idx = pd.IndexSlice
result = pd.DataFrame(index=range(1,11),columns=
             pd.MultiIndex.from_product([['mae', 'rmse', 'r2'],
                                         ['original', 'correct']]))


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
result.to_excel('/home/svu/e0560091/testingrelu_re_biadam.xlsx')


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
original_test_y = 10**original_test_y -1 # log scaling back for test_y
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
writer = pd.ExcelWriter('/home/svu/e0560091/testingrelu_cl_biadam.xlsx', engine = 'xlsxwriter')
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
