import lightgbm as lgb
import xgboost as xgb
import onnx
import tensorflow as tf
from environs import Env
from tf2onnx.tf_loader import from_saved_model
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost, convert_lightgbm, \
    convert_tensorflow


env = Env()
MODEL_PATH = env.str('MODEL_PATH', './models/lgb')
MODEL_TYPE = env.str('MODEL_TYPE', 'lgb')
ONNX_MODEL_PATH = env.str('ONNX_MODEL_PATH', './models/onnx')


initial_type = [('float_input', FloatTensorType([1, 39]))]
if MODEL_TYPE == 'xgb':
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    onx = convert_xgboost(model, initial_types=initial_type)
elif MODEL_TYPE == 'lgb':
    model = lgb.Booster(model_file=MODEL_PATH)
    onx = convert_lightgbm(model, initial_types=initial_type)
elif MODEL_TYPE == 'deepctr':
    graph_def, inputs, outputs = from_saved_model(MODEL_PATH, None, None)
    tf.compat.v1.disable_eager_execution()
    onx = convert_tensorflow(
        graph_def, input_names=inputs, output_names=outputs)
else:
    raise TypeError(f'模型类型错误：{MODEL_TYPE}')

onnx.save(onx, ONNX_MODEL_PATH)
