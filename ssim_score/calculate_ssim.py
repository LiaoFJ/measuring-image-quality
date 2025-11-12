import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image

import ssim_score

def resize_and_pad(img, target_height=1024, target_width=768):
    """将图像调整到目标尺寸，保持纵横比"""
    h, w = img.shape[-2:]

    # 计算缩放比例
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 先调整大小
    img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # 然后填充到目标尺寸
    pad_h = (target_height - new_h) // 2
    pad_w = (target_width - new_w) // 2
    pad_h_remainder = target_height - new_h - pad_h
    pad_w_remainder = target_width - new_w - pad_w

    img = F.pad(img, (pad_w, pad_w_remainder, pad_h, pad_h_remainder))
    return img

def image_to_tensor(img):
    img = img.convert('RGB')  # convert to RGB
    img_array = np.array(img)  # shape: (H, W, 3)
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1)  # change to (3, H, W)
    img_tensor = img_tensor.unsqueeze(0) / 255.0  # add batch dim and normalize
    return img_tensor

def get_ssim_score(real_dir, fake_dir):
    ssim_scores = []
    ssim_fn = ssim_score.SSIM(window_size=11)
    for original_path, generated_path in zip(os.listdir(real_dir), os.listdir(fake_dir)):
        original_img = Image.open(os.path.join(real_dir, original_path))
        generated_img = Image.open(os.path.join(fake_dir, generated_path))
        img1 = image_to_tensor(original_img).cuda()
        img2 = image_to_tensor(generated_img).cuda()

        # resize images
        img1 = resize_and_pad(img1)
        img2 = resize_and_pad(img2)

        score = ssim_fn(img1, img2).item()
        ssim_scores.append(score)
    print("Average SSIM:", sum(ssim_scores) / len(ssim_scores))
    return sum(ssim_scores) / len(ssim_scores)

if __name__ == '__main__':
    original_dir='/home/user/workspace/dataset/VTON-HD/test/image'
    generated_dir='/home/user/workspace/generated_images/vitonhd/paired/catvton/vitonhd-1024/paired'
    get_ssim_score(original_dir, generated_dir)