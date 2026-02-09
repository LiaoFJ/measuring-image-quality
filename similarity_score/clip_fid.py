import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from scipy import linalg
from cleanfid import fid

class CLIPFIDCalculator:
    """CLIP-FID 计算器"""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_features(self, image):
        """提取单张图像的CLIP特征"""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)

        return image_features.cpu().numpy()

    def extract_features_from_dir(self, image_dir):
        """从目录中提取所有图像的特征"""
        image_files = sorted([f for f in os.listdir(image_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

        features_list = []
        print(f"Extracting features from {len(image_files)} images in {image_dir}")

        for i, img_file in enumerate(image_files):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(image_files)}")

            try:
                img_path = os.path.join(image_dir, img_file)
                image = Image.open(img_path).convert('RGB')
                features = self.extract_features(image)
                features_list.append(features)
            except Exception as e:
                print(f"Warning: Failed to process {img_file}: {e}")
                continue

        return np.vstack(features_list)

    def calculate_statistics(self, features):
        """计算均值和协方差矩阵"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """计算FID分数"""
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                print(f"Warning: Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        return fid


def clip_fid_score_dir(real_dir, fake_dir, model_name="openai/clip-vit-base-patch32"):
    """
    计算两个目录之间的CLIP-FID分数

    Args:
        real_dir: 真实图像目录
        fake_dir: 生成图像目录
        model_name: CLIP模型名称

    Returns:
        CLIP-FID分数（越小越好）
    """
    calculator = CLIPFIDCalculator(model_name=model_name)

    # 提取特征
    print("\nExtracting features from real images...")
    real_features = calculator.extract_features_from_dir(real_dir)

    print("\nExtracting features from generated images...")
    fake_features = calculator.extract_features_from_dir(fake_dir)

    # 计算统计量
    print("\nCalculating statistics...")
    mu_real, sigma_real = calculator.calculate_statistics(real_features)
    mu_fake, sigma_fake = calculator.calculate_statistics(fake_features)

    # 计算FID
    print("\nCalculating CLIP-FID score...")
    fid_score = calculator.calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)

    return fid_score


if __name__ == '__main__':
    real_dir = '/home/user/workspace/dataset/VTON-HD/test/image'
    fake_dir = '/home/user/workspace/generated_images/vitonhd/paired/catvton/vitonhd-1024/paired'

    score = fid.compute_fid(real_dir, fake_dir, mode="clean", model_name="clip_vit_b_32")
    print(score)

    # score = clip_fid_score_dir(real_dir=real_dir, fake_dir=fake_dir)
    # print(f"\nCLIP-FID Score: {score:.4f}")