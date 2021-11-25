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

import config as config_module
from trainer.builder import build_transform
import torch
##################################################################

def onnx_model_check(model_path):
  model = onnx.load(model_path)
  onnx.checker.check_model(model)
  print("The model is checked!")


if __name__ == "__main__":
  onnx_path = "/home/muzi2045/Documents/project/OrienMask/onnx_export/orienmask_yolov3_fpn_sim.onnx"
  onnx_model_check(onnx_path)

  img_path =  "/home/muzi2045/Pictures/ros_output/yf_gc_day/frame0626.jpg"
  device = torch.device('cpu')

  config_name = "orienmask_yolo_coco_544_anchor4_fpn_plus_infer"
  config = getattr(config_module, config_name)
  config['transform']['use_cuda'] = False
  transform = build_transform(config['transform'])

  ### Start OnnxRuntime Infernece Session ###
  src_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
  src_image = torch.tensor(src_image, dtype=torch.float32).to(device)
  image = transform(src_image.unsqueeze(0))

  # print(type(image))
  input_image = image.cpu().numpy()

  print(f" #### {image.shape}")

  ort_sess = ort.InferenceSession(onnx_path)
  input = { "input" : (input_image)}
  output = ort_sess.run(None, input)

  print(f"### {output[0].shape}")
  bbox32 = output[0]
  np.savetxt("orienmask_bbox32_ort.txt",
                  bbox32.reshape((-1, 255)),
                  fmt="%.6f",
                  delimiter=',')




  # print(f"### {type(output[1])}")
  # print(f"### {type(output[2])}")
  # print(f"### {type(output[3])}")
  # print(f"### {type(output[4])}")
  # print(f"### {type(output[5])}")

########################################### 