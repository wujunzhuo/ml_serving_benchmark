import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from deepctr.feature_column import DenseFeat, SparseFeat
from deepctr.models.dcn import DCN
from environs import Env
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split


env = Env()
DATA_PATH = env.str('DATA_PATH', './data/train_1m.txt')
DATA_NROWS = env.int('DATA_NROWS', 100000)
RANDOM_SEED = env.int('RANDOM_SEED', 2020)
TRANS_PATH = env.str('TRANS_PATH_IMPUTER', './outputs/trans')
MODEL_PATH_XGB = env.str('MODEL_PATH_XGB', './outputs/xgb')
MODEL_PATH_LGB = env.str('MODEL_PATH_LGB', './outputs/lgb')
MODEL_PATH_DCN = env.str('MODEL_PATH_DCN', './outputs/dcn')

label = 'label'
num_feat = ['I' + str(i) for i in range(1, 14)]
cat_feat = ['C' + str(i) for i in range(1, 27)]

print('读数据')
df = pd.read_csv(
    DATA_PATH, delimiter='\t', header=None, nrows=DATA_NROWS,
    keep_default_na=False, na_values='',
    names=[label] + num_feat + cat_feat)
df[cat_feat] = df[cat_feat].fillna('')

print('拆分训练/验证集')
train_df, valid_df = train_test_split(
    df, test_size=0.2, shuffle=True, random_state=RANDOM_SEED)
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)
train_X = train_df.loc[:, num_feat + cat_feat]
valid_X = valid_df.loc[:, num_feat + cat_feat]

print('数据变换')
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler()),
])
cat_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1)),
])
trans = ColumnTransformer(
    transformers=[
        ('num', num_transformer, list(range(13))),
        ('cat', cat_transformer, list(range(13, 39)))
    ])
train_X[num_feat + cat_feat] = trans.fit_transform(train_X)
valid_X[num_feat + cat_feat] = trans.transform(valid_X)
joblib.dump(trans, TRANS_PATH)

train_X[cat_feat] = train_X[cat_feat].astype(np.int32) + 1
valid_X[cat_feat] = valid_X[cat_feat].astype(np.int32) + 1

print('XGBoost模型')
model = xgb.XGBClassifier(
    use_label_encoder=False, random_state=RANDOM_SEED,
    n_estimators=200, max_depth=5)
model.fit(
    train_X, train_df[label],
    eval_set=[(valid_X, valid_df[label])],
    eval_metric=['auc'], early_stopping_rounds=None)
model.save_model(MODEL_PATH_XGB)

print('LightGBM模型')
model = lgb.LGBMClassifier(n_estimators=200, random_state=RANDOM_SEED)
model.fit(
    train_X, train_df[label],
    eval_set=[(valid_X, valid_df[label])],
    eval_metric=['auc'], early_stopping_rounds=None)
model.booster_.save_model(MODEL_PATH_LGB)

print('DCN模型')
feature_columns = [
    DenseFeat(c, 1) for c in num_feat] + [
    SparseFeat(c, n + 1, 'auto') for c, n in train_X[cat_feat].max().items()
]
model = DCN(
    feature_columns, feature_columns, cross_num=4, dnn_use_bn=True,
    cross_parameterization='matrix', seed=RANDOM_SEED)
model.compile('adam', 'binary_crossentropy', metrics=['AUC'])
model.fit(
    [x for _, x in train_X.items()], train_df[label],
    validation_data=([x for _, x in valid_X.items()], valid_df[label]),
    shuffle=False, epochs=1)
model.save(MODEL_PATH_DCN)
