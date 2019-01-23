import onnx
import onnx.tools.net_drawer
model = onnx.load("G.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))