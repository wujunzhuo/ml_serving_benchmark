import numpy as np
import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf
from environs import Env
from flask import Flask, request, jsonify
from deepctr.layers import custom_objects


app = Flask(__name__)
env = Env()
MODEL_PATH = env.str('MODEL_PATH', './models/xgb')
MODEL_TYPE = env.str('MODEL_TYPE', 'xgb')

if MODEL_TYPE == 'xgb':
    model = xgb.Booster(model_file=MODEL_PATH)
elif MODEL_TYPE == 'lgb':
    model = lgb.Booster(model_file=MODEL_PATH)
elif MODEL_TYPE == 'deepctr':
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects)
else:
    raise TypeError(f'模型类型错误：{MODEL_TYPE}')


@app.route('/predict', methods=['POST'])
def predict():
    req = request.json
    data = np.array(req['inputs'])

    if MODEL_TYPE == 'xgb':
        pred = model.predict(xgb.DMatrix(data))
    elif MODEL_TYPE == 'lgb':
        pred = model.predict(data)
    else:
        pred = model.predict([data[:, i] for i in range(data.shape[1])])

    return jsonify(pred=pred.tolist())
