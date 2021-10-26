import sys
import argparse
import os
import struct
import torch

import model as model_module
import config as config_module

from trainer.builder import build, build_transform, build_postprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='inference config (default: None)')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-o', '--output', help='Output (.wts) file path (optional)')
    args = parser.parse_args() 
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.splitext(os.path.basename(args.weights))[0] + '.wts')
    return args.config, args.weights, args.output

config_file, pt_file, wts_file = parse_args()

# print(f"{ config_file } -- { pt_file } -- { wts_file }")

config = getattr(config_module, config_file)


# Initialize
device = torch.device('cuda:0')
config['model']['pretrained'] = None
model = build(config['model'], model_module).to(device)
weights = torch.load(pt_file, map_location=device)
weights = weights['state_dict'] if 'state_dict' in weights else weights
model.load_state_dict(weights, strict=True)

model.eval()

# print(model)


with open(wts_file, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f' ,float(vv)).hex())
        f.write('\n')
