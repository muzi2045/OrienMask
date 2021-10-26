#! /usr/bin/python3

############### import lib #####################################
import sys
import onnx
import numpy as np
import onnxruntime as ort
path = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if path in sys.path:
    sys.path.remove(path)
import cv2

##################################################################

def onnx_model_check(model_path):
  model = onnx.load(model_path)
  graph = model.graph
  # nodes = graph.node
  # count = 0
  # for node in nodes:
  #   if not node.name:
  #     node.name = "rand_node_name_" + str(count)
  #     count = count + 1
  onnx.checker.check_model(model)
  print("The model is checked!")


if __name__ == "__main__":

  onnx_path = "/home/muzi2045/Documents/project/OrienMask/onnx_export/orienmask_yolov3_fpn_sim.onnx"

  img_path = "/home/muzi2045/Pictures/ros_output/yf_gc_day/frame0626.jpg"

  # onnx_model_check(onnx_path)

  ### Start OnnxRuntime Infernece Session ###

  src_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

  print(type(src_image))

  # ort_sess = ort.InferenceSession(onnx_path)

  # input = {ort_sess.get_inputs()[0].name: (input_image)}

  # output = ort_sess.run(None, input)

  # print(type(output))
  # final_output = np.array(output[0])


  ########################################### 