import onnx
from onnx import helper
from torchreid.utils.feature_extractor import FeatureExtractor
# 加载 ONNX 模型
onnx_model_path = '/home/spring/test_qzj/project/deep-person-reid-master/base/version17_no_test/osnet_x1_0_90.onnx'
onnx_model = onnx.load(onnx_model_path)

# # 打印 ONNX 模型结构
# model_structure = helper.printable_graph(onnx_model.graph)
# print(model_structure)  # 打印模型结构
