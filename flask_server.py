import numpy as np
import lightgbm as lgb
import xgboost as xgb
# import tensorflow as tf
from environs import Env
from flask import Flask, request, jsonify
# from deepctr.layers import custom_objects


app = Flask(__name__)
env = Env()
model_path = env.str('MODEL_PATH', './models/xgb')
model_type = env.str('MODEL_TYPE', 'xgb')

if model_type == 'xgb':
    model = xgb.Booster(model_file=model_path)
elif model_type == 'lgb':
    model = lgb.Booster(model_file=model_path)
# elif model_type == 'deepctr':
#     model = tf.keras.models.load_model(model_path, custom_objects)
else:
    raise TypeError(f'模型类型错误：{model_type}')


@app.route('/predict', methods=['POST'])
def predict():
    data = np.array(request.json['inputs'])
    if model_type == 'xgb':
        pred = model.predict(xgb.DMatrix(data)).tolist()
    elif model_type == 'lgb':
        pred = model.predict(data).tolist()
    else:
        pred = None
    return jsonify(pred=pred)
