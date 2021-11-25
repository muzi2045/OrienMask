#!/usr/bin/python3
from typing_extensions import final
from trainer.builder import build, build_transform, build_postprocess
import model as model_module
import config as config_module
import utils.visualizer as visualizer_module
import numpy as np

import torch
import json
import argparse
import sys

path = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if path in sys.path:
    sys.path.remove(path)
import cv2
import math
import matplotlib.pyplot as plt

import torch.nn.functional as F


def pad(image, size_divisor=32, pad_value=0):
    height, width = image.shape[-2:]
    new_height = int(math.ceil(height / size_divisor) * size_divisor)
    new_width = int(math.ceil(width / size_divisor) * size_divisor)
    pad_left, pad_top = (new_width - width) // 2, (new_height - height) // 2
    pad_right, pad_down = new_width - width - \
        pad_left, new_height - height - pad_top

    padding = [pad_left, pad_right, pad_top, pad_down]
    image = F.pad(image, padding, value=pad_value)
    pad_info = padding + [new_height, new_width]

    return image, pad_info


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        help='inference config (default: None)')
    parser.add_argument('-w',
                        '--weights',
                        default=None,
                        type=str,
                        help='model weights to inference (default: None)')
    args = parser.parse_args()

    config_name = "orienmask_yolo_coco_544_anchor4_fpn_plus_infer"
    weight_path = "checkpoints/OrienMaskAnchor4FPNPlus/orienmask_yolo.pth"
    # Load config
    config = getattr(config_module, config_name)

    # Device
    use_cuda = config['n_gpu'] > 0
    if use_cuda:
        device = torch.device('cuda:0')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', True)
    else:
        device = torch.device('cpu')

    # Build model, transform and postprocess
    config['model']['pretrained'] = None
    model = build(config['model'], model_module).to(device)
    weights = torch.load(weight_path, map_location=device)
    weights = weights['state_dict'] if 'state_dict' in weights else weights
    model.load_state_dict(weights, strict=True)
    config['transform']['use_cuda'] = use_cuda
    transform = build_transform(config['transform'])
    postprocess = build_postprocess(config['postprocess'], device=device)
    visualizer = build(config['visualizer'], visualizer_module, device=device)

    img_path = "/home/muzi2045/Documents/project/tensorrtx/orienmask/samples/test.jpg"

    with torch.no_grad():
        model.eval()
        original_image = cv2.imread(img_path)
        original_image = torch.tensor(original_image,
                                      dtype=torch.float32).to(device)
        
        src_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        src_image = torch.tensor(src_image, dtype=torch.float32).to(device)
        image = transform(src_image.unsqueeze(0))
        # print(image.shape)
        # print(type(image))
      
        image, pad_info = pad(image)
        # print(f"padded image size: {image.shape}")
        # out_img = image.permute(0, 2, 3, 1).cpu().numpy()
        # np.savetxt("image_input_padded.txt",
        #            out_img.reshape((-1, 3)),
        #            fmt="%.6f",
        #            delimiter=',')

        prediction = model(image)
        bbox32 = prediction[0][0]
        orien32 = prediction[0][1]

        print(f"bbox32 size: {bbox32.shape}")
        print(f"orien32 size: {orien32.shape}")

        bbox32 = bbox32.permute(0, 2, 3, 1).cpu().numpy()
        orien32 = orien32.permute(0, 2, 3, 1).cpu().numpy()

        np.savetxt("orienmask_bbox32.txt",
                   bbox32.reshape((-1, 255)),
                   fmt="%.6f",
                   delimiter=',')

        np.savetxt("orienmask_orien32.txt",
            orien32.reshape((-1, 6)),
            fmt="%.6f",
            delimiter=',')
        final_output = postprocess(prediction)
        show_image = visualizer(final_output[0], original_image, pad_info)
        # print(f" bbox shape: {final_output[0]['bbox'].shape}")
        # print(f" mask shape: {final_output[0]['mask'].shape}")
        # print(f" cls shape: {final_output[0]['cls'].shape}")
        cv2.imshow("test", show_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
