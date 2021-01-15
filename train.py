import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from category_encoders import OrdinalEncoder
from deepctr.feature_column import DenseFeat, SparseFeat
from deepctr.models.dcn import DCN
from environs import Env
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


env = Env()
DATA_PATH = env.str('DATA_PATH', './data/train_1m.txt')
DATA_NROWS = env.int('DATA_NROWS', 100000)
RANDOM_SEED = env.int('RANDOM_SEED', 2020)
MODEL_PATH_XGB = env.str('MODEL_PATH_XGB', './models/xgb')
MODEL_PATH_LGB = env.str('MODEL_PATH_LGB', './models/lgb')
MODEL_PATH_DCN = env.str('MODEL_PATH_DCN', './models/dcn')

label = 'label'
dense_feat = ['I' + str(i) for i in range(1, 14)]
sparse_feat = ['C' + str(i) for i in range(1, 27)]

print('读数据')
df = pd.read_csv(
    DATA_PATH, delimiter='\t', header=None, nrows=DATA_NROWS,
    names=[label] + dense_feat + sparse_feat)

print('拆分训练/验证集')
train_df, valid_df = train_test_split(
    df, test_size=0.2, shuffle=True, random_state=RANDOM_SEED)
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)
train_X = train_df.loc[:, dense_feat + sparse_feat]
train_y = train_df.loc[:, label]
valid_X = valid_df.loc[:, dense_feat + sparse_feat]
valid_y = valid_df.loc[:, label]


print('缺失值填充')
imputer = SimpleImputer(strategy='mean')
train_X[dense_feat] = imputer.fit_transform(train_X[dense_feat])
valid_X[dense_feat] = imputer.transform(valid_X[dense_feat])
train_X[sparse_feat] = train_X[sparse_feat].fillna('')
valid_X[sparse_feat] = valid_X[sparse_feat].fillna('')

print('连续型变量数值归一化')
scaler = MinMaxScaler()
train_X[dense_feat] = scaler.fit_transform(train_X[dense_feat])
valid_X[dense_feat] = scaler.transform(valid_X[dense_feat])

print('离散型变量字典编码')
enc = OrdinalEncoder(cols=sparse_feat)
train_X[sparse_feat] = enc.fit_transform(train_X[sparse_feat]) + 1
valid_X[sparse_feat] = enc.transform(valid_X[sparse_feat]) + 1

print('XGBoost模型')
model = xgb.XGBClassifier(
    use_label_encoder=False, random_state=RANDOM_SEED,
    n_estimators=200, max_depth=5)
model.fit(
    train_X, train_y, eval_set=[(valid_X, valid_y)],
    eval_metric=['auc'], early_stopping_rounds=None)
model.save_model(MODEL_PATH_XGB)

print('LightGBM模型')
model = lgb.LGBMClassifier(n_estimators=200, random_state=RANDOM_SEED)
model.fit(
    train_X, train_y, eval_set=[(valid_X, valid_y)],
    eval_metric=['auc'], early_stopping_rounds=None)
model.booster_.save_model(MODEL_PATH_LGB)

print('DCN模型')
feature_columns = [
    DenseFeat(c, 1) for c in dense_feat] + [
    SparseFeat(c, n + 1, 'auto') for c, n in train_X[sparse_feat].max().items()
]
model = DCN(
    feature_columns, feature_columns, cross_num=4, dnn_use_bn=True,
    cross_parameterization='matrix', seed=RANDOM_SEED)
model.compile('adam', 'binary_crossentropy', metrics=['AUC'])
train_X_list = [train_X[c] for c in dense_feat + sparse_feat]
valid_X_list = [valid_X[c] for c in dense_feat + sparse_feat]
model.fit(
    train_X_list, train_y, validation_data=(valid_X_list, valid_y),
    shuffle=False, epochs=1)
model.save(MODEL_PATH_DCN)
