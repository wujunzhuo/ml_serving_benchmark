import joblib
import lightgbm as lgb
import xgboost as xgb
import onnx
import tensorflow as tf
from environs import Env
from skl2onnx import convert_sklearn
from tf2onnx.tf_loader import from_saved_model
from onnxconverter_common.data_types import FloatTensorType, StringTensorType
from onnxmltools.convert import convert_xgboost, convert_lightgbm, \
    convert_tensorflow


env = Env()
TRANS_PATH = env.str('TRANS_PATH', './outputs/trans')
MODEL_PATH = env.str('MODEL_PATH', './outputs/lgb')
MODEL_TYPE = env.str('MODEL_TYPE', 'lgb')
ONNX_TRANS_PATH = env.str('TRANS_PATH', './outputs/trans.onnx')
ONNX_MODEl_PATH = env.str('ONNX_PATH', './outputs/model.onnx')


trans_initial_type = [
    ('num_feat', FloatTensorType([None, 13])),
    ('cat_feat', StringTensorType([None, 26]))
]
model_initial_type = [('feat', FloatTensorType([None, 39]))]


trans = joblib.load(TRANS_PATH)
onx = convert_sklearn(trans, initial_types=trans_initial_type)
onnx.save(onx, ONNX_TRANS_PATH)


if MODEL_TYPE == 'xgb':
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    onx = convert_xgboost(model, initial_types=model_initial_type)
elif MODEL_TYPE == 'lgb':
    model = lgb.Booster(model_file=MODEL_PATH)
    onx = convert_lightgbm(model, initial_types=model_initial_type)
elif MODEL_TYPE == 'deepctr':
    graph_def, inputs, outputs = from_saved_model(MODEL_PATH, None, None)
    tf.compat.v1.disable_eager_execution()
    onx = convert_tensorflow(
        graph_def, input_names=inputs, output_names=outputs)
else:
    raise TypeError(f'模型类型错误：{MODEL_TYPE}')

onnx.save(onx, ONNX_MODEl_PATH)
