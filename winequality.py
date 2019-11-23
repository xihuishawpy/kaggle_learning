
import pandas as pd 
import numpy as np 
import lightgbm as lgb  
from sklearn.preprocessing import LabelEncoder
from itertools import combinations



train_df = pd.read_csv('train.csv',header=None , sep=';')
test_df = pd.read_csv('test.csv',header=None , sep=';')
print('train_df.shape',train_df.shape,'test_df.shape',test_df.shape)

train_df = train_df[train_df[11]!='quality']
train_df.shape

lbl = LabelEncoder().fit(train_df[11])
train_df[11] = lbl.transform(train_df[11])
train_df.dtypes

# 转浮点数 ，方便计算
train_df_labl = train_df[[0,1,2,3,4,5,6,7,8,9,10]].astype('float')
test_df = test_df.astype('float')


# combinations 自由组合成元组  
# 目的：构造不同特征组合（两两组合）  np.log1p() == log(1+x)
for a ,b in combinations([0,1,2,3,4,7,8,9,10],2):
    train_df_labl[str(a)+'_'+str(b)] = train_df_labl[a]+train_df_labl[b]
    train_df_labl[str(a)+'/'+str(b)] = train_df_labl[a] / train_df_labl[b]
    train_df_labl[str(a)+'*'+str(b)] = train_df_labl[a] * train_df_labl[b]
    train_df_labl[str(a)+'/log'+str(b)] = train_df_labl[a] / np.log1p(train_df_labl[b])

    test_df[str(a)+'_'+str(b)] = test_df[a]+test_df[b]
    test_df[str(a)+'/'+str(b)] = test_df[a] / test_df[b]
    test_df[str(a)+'*'+str(b)] = test_df[a] * test_df[b]
    test_df[str(a)+'/log'+str(b)] = test_df[a] / np.log1p(test_df[b])

train_df = pd.concat([train_df_labl,train_df[11]],axis=1)

train_df.head()


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error 

# StratifiedKFold用法类似Kfold，
# 但 StratifiedKFold是分层采样，根据标签中不同类别占比来进行拆分数据的。
# 确保训练集，测试集中各类别样本的比例与原始数据集中相同。

n_fold = 10 
kf = StratifiedKFold(n_splits = n_fold , shuffle =True)
eval_fun = mean_absolute_error




def run_oof(clf, X_train, y_train, X_test, kf):
    print(clf)
    preds_train = np.zeros((len(X_train),7),dtype= np.float)
    preds_test = np.zeros((len(X_test),7),dtype= np.float)
    train_loss =[] ; test_loss=[]
    
    # 分层采样
    i =1 
    for train_index, test_index in kf.split(X_train, y_train):
        x_tr = X_train[train_index]
        x_te = X_train[test_index]
        y_tr = y_train[train_index]
        y_te = y_train[test_index]

        clf.fit(x_tr ,y_tr ,eval_set=[(x_te,y_te)],early_stopping_rounds=500,verbose=False)

        train_loss.append(eval_fun(y_tr, np.argmax(clf.predict_proba(x_tr)[:], 1)))
        test_loss.append(eval_fun(y_te, np.argmax(clf.predict_proba(x_te)[:], 1)))

        preds_train[test_index] = clf.predict_proba(x_te)[:]
        preds_test += clf.predict_proba(X_test)[:]
         
        # 每层样本结果
        print('{0}: Train {1:0.7f} Val {2:0.7f}/{3:0.7f}'.format(i, train_loss[-1], test_loss[-1], np.mean(test_loss)))
        print('-' * 50)
        i+=1

    print('Train:',train_loss)
    print('Val:',test_loss)
    print('-' * 50)
    
    # 测试集上对每种特征预测的概率 平均表现
    preds_test /= n_fold

    return preds_train ,preds_test



params = {
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'max_depth': 5,
    'lambda_l1': 2,
    'boosting': 'gbdt',
    'objective': 'multiclass',
    'n_estimators': 3000,
    'metric': 'multi_error',
    'num_class': 7,
    'feature_fraction': .75,
    'bagging_fraction': .85,
    'seed': 99,
    'num_threads': 20,
    'verbose': -1
}


train_pred, test_pred = run_oof(lgb.LGBMClassifier(**params), 
                                train_df.drop(11, axis=1).values, 
                                train_df[11].values, 
                                test_df.values, 
                                kf)



submit = pd.DataFrame()
submit[0] = range(len(test_df))
submit[1] = lbl.inverse_transform(np.argmax(test_pred, 1))
submit.to_csv('lgb.csv', index=None, header=None)





        


























