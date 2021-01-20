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
MODEL_PATH_XGB = env.str('MODEL_PATH_XGB', './outputs/xgb')
MODEL_PATH_LGB = env.str('MODEL_PATH_LGB', './outputs/lgb')
MODEL_PATH_DCN = env.str('MODEL_PATH_DCN', './outputs/dcn')
ONNX_TRANS_PATH = env.str('TRANS_PATH', './outputs/trans.onnx')
ONNX_MODEl_PATH_XGB = env.str('ONNX_MODEl_PATH_XGB', './outputs/xgb.onnx')
ONNX_MODEl_PATH_LGB = env.str('ONNX_MODEl_PATH_LGB', './outputs/lgb.onnx')
ONNX_MODEl_PATH_DCN = env.str('ONNX_MODEl_PATH_DCN', './outputs/dcn.onnx')


trans_initial_type = [
    ('num_feat', FloatTensorType([None, 13])),
    ('cat_feat', StringTensorType([None, 26]))
]
model_initial_type = [('num_feat', FloatTensorType([None, 39]))]


print('convert sklearn transformer')
trans = joblib.load(TRANS_PATH)
onx = convert_sklearn(trans, initial_types=trans_initial_type)
onnx.save(onx, ONNX_TRANS_PATH)


print('convert XGBoost model')
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH_XGB)
onx = convert_xgboost(model, initial_types=model_initial_type)
onnx.save(onx, ONNX_MODEl_PATH_XGB)


print('convert LightGBM model')
model = lgb.Booster(model_file=MODEL_PATH_LGB)
onx = convert_lightgbm(model, initial_types=model_initial_type)
onnx.save(onx, ONNX_MODEl_PATH_LGB)


print('convert DCN model')
graph_def, inputs, outputs = from_saved_model(MODEL_PATH_DCN, None, None)
tf.compat.v1.disable_eager_execution()
onx = convert_tensorflow(graph_def, input_names=inputs, output_names=outputs)
onnx.save(onx, ONNX_MODEl_PATH_DCN)
