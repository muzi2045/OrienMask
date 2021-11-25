
from trainer.builder import build, build_transform, build_postprocess
import model as model_module
import config as config_module

import torch
import json
import argparse

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='inference config (default: None)')
    parser.add_argument('-w', '--weights', default=None, type=str,
                        help='model weights to inference (default: None)')
    args = parser.parse_args()

    # Load config
    if args.config.endswith('.json'):
        config = json.load(open(args.config))
    else:
        config = getattr(config_module, args.config)

    # Device
    # use_cuda = config['n_gpu'] > 0
    # if use_cuda:
    #     device = torch.device('cuda:0')
    #     torch.backends.cudnn.enabled = True
    #     torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', True)
    # else:
    device = torch.device('cpu')

    # Build model, transform and postprocess
    config['model']['pretrained'] = None
    model = build(config['model'], model_module).to(device)
    weights = torch.load(args.weights, map_location=device)
    weights = weights['state_dict'] if 'state_dict' in weights else weights
    model.load_state_dict(weights, strict=True)
    # config['transform']['use_cuda'] = use_cuda
    # transform = build_transform(config['transform'])
    postprocess = build_postprocess(config['postprocess'], device=device)

    with torch.no_grad():
        model.eval()
        input = torch.randn(1, 3, 544, 544, device='cpu')

        input_names = ["input"]
        inputs = [input]
        inputs = tuple(inputs)

        predictions = model(input)

        print(type(predictions))
        print(f" {predictions[0][0].shape}")
        print(f" {predictions[0][1].shape}")

        ### there has three head output from model
        ## (bbox32, orien32) --> ([1, 255, 17, 17], [1, 6, 136, 136])
        #  (bbox16, orien16) --> ([1, 255, 34, 34], [1, 6, 136, 136])
        #  (bbox8, orien8)   --> ([1, 255, 68, 68], [1, 6, 136, 136])
        torch.onnx.export(model, inputs,  "./onnx_export/orienmask_yolov3_fpn_opset12.onnx", verbose=False,
                    input_names=input_names, export_params=True, keep_initializers_as_inputs=True, opset_version=12)
        # predictions = postprocess(predictions)[0]

        # print(f" bbox shape: {predictions['bbox'].shape}")
        # print(f" mask shape: {predictions['mask'].shape}")
        # print(f" cls shape: {predictions['cls'].shape}")

        