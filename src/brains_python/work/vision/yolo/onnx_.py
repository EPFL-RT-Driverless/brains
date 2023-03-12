import onnx_graphsurgeon as gs
import onnx
import numpy as np

print("Running BatchedNMSDynamic_TRT plugin the ONNX model.. ")

input_model_path = "./weights/yolov7-tiny.onnx"
output_model_path = "./weights/yolov7-tiny-cat.onnx"

graph = gs.import_onnx(onnx.load(input_model_path))
# graph.inputs[0].shape=[1,3,1024,544]

tmap = graph.tensors()

out1, out2, out3 = tmap["411"], tmap["477"], tmap["543"]

print(out1.shape)
print(out2.shape)
print(out3.shape)


node1_var = gs.Variable(
    name="out1_reshaped", dtype=np.float32, shape=(1, 26112, 30)
)  # 3*128*68
node1_shape = gs.Variable(name="node1_shape", dtype=np.int32).to_constant(
    np.array([1, -1, 30]).astype(np.int32)
)

node2_var = gs.Variable(
    name="out2_reshaped", dtype=np.float32, shape=(1, 6528, 30)
)  # 3*64*34
node2_shape = gs.Variable(name="node2_shape", dtype=np.int32).to_constant(
    np.array([1, -1, 30]).astype(np.int32)
)

node3_var = gs.Variable(
    name="out3_reshaped", dtype=np.float32, shape=(1, 1632, 30)
)  # 3*32*17
node3_shape = gs.Variable(name="node3_shape", dtype=np.int32).to_constant(
    np.array([1, -1, 30]).astype(np.int32)
)

node1 = gs.Node(
    name="Reshape_out1",
    op="Reshape",
    attrs={"allowzero": 0},
    inputs=[out1, node1_shape],
    outputs=[node1_var],
)
node2 = gs.Node(
    name="Reshape_out1",
    op="Reshape",
    attrs={"allowzero": 0},
    inputs=[out2, node2_shape],
    outputs=[node2_var],
)
node3 = gs.Node(
    name="Reshape_out1",
    op="Reshape",
    attrs={"allowzero": 0},
    inputs=[out3, node3_shape],
    outputs=[node3_var],
)

out_var = gs.Variable(
    name="output0", dtype=np.float32, shape=(1, 34272, 30)
)  # 26112 + 6528 + 1632
output = gs.Node(
    name="concat_out",
    op="Concat",
    attrs={"axis": 1},
    inputs=[node1_var, node2_var, node3_var],
    outputs=[out_var],
)

graph.nodes.append(node1)
graph.nodes.append(node2)
graph.nodes.append(node3)
graph.nodes.append(output)
graph.outputs = [out_var]

# Disconnect old subgraph
# out1.inputs.clear()
# out2.inputs.clear()
# out3.inputs.clear()

# NOTE: Need to cleanup to remove the old NMS node properly.
# Finally, we can save the model.
# graph.cleanup().toposort()
onnx.save_model(gs.export_onnx(graph), output_model_path)


# import onnx_graphsurgeon as gs
# import onnx
# import numpy as np

# print ("Running BatchedNMSDynamic_TRT plugin the ONNX model.. ")

# input_model_path = "./weights/yolov7-tiny-cones-1200.onnx"
# output_model_path = "./weights/yolov7-tiny-cones-1200-gs.onnx"

# graph = gs.import_onnx(onnx.load(input_model_path))
# # graph.inputs[0].shape=[1,3,1024,544]

# tmap = graph.tensors()
# # NOTE: Input and output tensors are model-dependent.
# # From your logging output, it looks like these are the ones of interest:
# #     input: "yolonms_layer_1/ExpandDims_1:0"
# #     input: "yolonms_layer_1/ExpandDims_3:0"
# #     output: "casted"
# # The other input tensors turn into plugin attributes (see `attrs` below)
# boxes, scores, nms_out = tmap["onnx::NonMaxSuppression_1915"], tmap["onnx::NonMaxSuppression_1919"], tmap["onnx::Gather_1922"]

# # Disconnect old subgraph
# boxes.outputs.clear()
# scores.outputs.clear()
# nms_out.inputs.clear()

# attrs = {
#     "keepTopK": 50, # Based on max_output_boxes_per_class
#     "iouThreshold": 0.5,
#     "scoreThreshold": 0.6,
#     "numClasses": 4,
#     "shareLoaction": False,
#     # TODO: Fill out any other attributes you may need
#     # (see https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin#parameters)
# }

# print (boxes.shape)
# print (scores.shape)
# node = gs.Node(op="BatchedNMS_TRT", attrs=attrs,
#                inputs=[boxes, scores], outputs=[nms_out])
# graph.nodes.append(node)
# print (boxes.shape)
# print(graph)
# # NOTE: Need to cleanup to remove the old NMS node properly.
# # Finally, we can save the model.
# graph.cleanup().toposort()
# onnx.save_model(gs.export_onnx(graph), output_model_path)
