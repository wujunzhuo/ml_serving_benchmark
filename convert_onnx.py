import lightgbm as lgb
import onnx
from environs import Env
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_lightgbm


env = Env()
MODEL_PATH_LGB = env.str('MODEL_PATH_LGB', './models/lgb')
ONNX_MODEL_PATH = env.str('ONNX_MODEL_PATH', './models/onnx')


model = lgb.Booster(model_file=MODEL_PATH_LGB)

initial_type = [('float_input', FloatTensorType([1, 39]))]
onx = convert_lightgbm(model, initial_types=initial_type)
onnx.save(onx, ONNX_MODEL_PATH)
