#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: get_encoder.py
@author:zyl
@contact:yilan.zhang@kaust.edu.sa
@time:12/23/24 9:28 PM
'''

import os
import torch
from torchvision import transforms
import timm
# import argparse
# torch.serialization.add_safe_globals([argparse.Namespace])

def load_checkpoint_remap_keys(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:  # 可能权重文件是 {'model': state_dict, ...}
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # 处理 key 前缀，例如 'image_encoder.model.cls_token' -> 'cls_token'
    new_state_dict = {}
    for key in state_dict.keys():
        if not key.startswith("image_encoder"):
            continue
        new_key = key.replace("image_encoder.model.", "")  # 去掉前缀
        new_state_dict[new_key] = state_dict[key]

    # 加载修改后的权重
    model.load_state_dict(new_state_dict, strict=True)  # 使用 strict=False 以防某些 key 仍然不匹配
    return model

def get_encoder(checkpoint_path, device=None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    # model.load_state_dict(torch.load(checkpoint_path, map_location="cpu",weights_only=False)["model"], strict=True)
    # 加载修正后的 checkpoint
    model = load_checkpoint_remap_keys(checkpoint_path, model)
    model.eval()
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return model, transform


if __name__ == "__main__":
    model, transform = get_encoder("/home/zhany0x/Documents/data/Pathology/checkpoints/pathomics/past_latest_2.pth")
    print(model)
    print(transform)
