import onnx
from onnx import version_converter, checker

base_path = "/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/251105-225225_osnet_pcb_512d_ibn_0.003_id124_pcb6-left-triplet-mark1501/osnet_pcb_512d_ibn-triplet-pre_False_id124_pcb6-left-triplet/model"
src_path = base_path + "/" + "osnet_pcb_512d_ibn.sim.onnx"
dst_path = base_path + "/" + "op_11.sim.onnx"

m = onnx.load(src_path)
m11 = version_converter.convert_version(m, 11)  # 关键一步：降版本到 11
checker.check_model(m11)                        # 校验图合法性
onnx.save(m11, dst_path)
print("Saved:", dst_path)

