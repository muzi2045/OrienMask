#! /usr/bin/env python3

from threading import main_thread
import rospy
import ros_numpy
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image

import sys
import os
import pickle
import shutil
import torch
import time

path = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if path in sys.path:
    sys.path.remove(path)
import cv2

bridge = CvBridge()

class Processor_ROS:
  def __init__(self) -> None:
    pass
  
  def read_config(self) -> None:
    pass

  def run(self, img):
    pass


def show_img(img):
  cv2.show("test", img)
  cv2.waitKey(3)

def img_callback(msg):

  rospy.loginfo(msg.header)

  cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")

  show_img(cv_image)


if __name__ == '__main__':
  

  config_path = "xxxxxx"
  model_path = "xxxxxx"

  proc = Processor_ROS()

  rospy.init_node("orienmask_node")

  img_sub = rospy.Subscriber("/front/usb_cam/image_raw", Image, img_callback, queue_size=1, buff_size=2**24)

  pub = rospy.Publisher("/img_processed", Image, queue_size=1)

  print("[+] OrienMask ROS Node has started! ")
  rospy.spin()
