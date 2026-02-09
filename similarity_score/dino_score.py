import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import numpy as np


class DINOSimilarity:
    def __init__(self, model_name="facebook/dinov2-base"):
        """
        初始化DINO模型用于计算图片相似度

        Args:
            model_name: 预训练模型名称，默认使用dinov2-base
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载模型和处理器
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_features(self, image_path):
        """
        提取单张图片的特征向量

        Args:
            image_path: 图片路径

        Returns:
            特征向量 (torch.Tensor)
        """
        # 加载并预处理图片
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用CLS token的特征作为图片的表示
            features = outputs.last_hidden_state[:, 0, :]

        return features

    def cosine_similarity(self, features1, features2):
        """
        计算两个特征向量的余弦相似度

        Args:
            features1: 第一张图片的特征向量
            features2: 第二张图片的特征向量

        Returns:
            余弦相似度分数 (float)
        """
        # 使用PyTorch的cosine_similarity函数
        similarity = F.cosine_similarity(features1, features2)
        return similarity.item()

    def calculate_similarity(self, image_path1, image_path2):
        """
        计算两张图片的余弦相似度

        Args:
            image_path1: 第一张图片路径
            image_path2: 第二张图片路径

        Returns:
            余弦相似度分数 (float)，范围[-1, 1]，越接近1表示越相似
        """
        # 提取两张图片的特征
        features1 = self.extract_features(image_path1)
        features2 = self.extract_features(image_path2)

        # 计算余弦相似度
        similarity = self.cosine_similarity(features1, features2)

        return similarity

def get_avg_dino_score(image1_dir, image2_dir):
    """
    计算多对图片的平均DINO相似度分数

    Args:
        image_pairs: 包含图片路径对的列表 [(img1_path, img2_path), ...]

    Returns:
        平均DINO相似度分数 (float)
    """
    dino_sim = DINOSimilarity(model_name="facebook/dinov2-base")
    scores = []

    # 获取并排序图像文件
    image_files1 = sorted([f for f in os.listdir(image1_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    image_files2 = sorted([f for f in os.listdir(image2_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    if len(image_files1) != len(image_files2):
        raise ValueError(f"目录中的图像数量不匹配: {len(image_files1)} vs {len(image_files2)}")

    print(f"Processing {len(image_files1)} image pairs...")

    for i, (img1_name, img2_name) in enumerate(zip(image_files1, image_files2)):

        img1_path = os.path.join(image1_dir, img1_name)
        img2_path = os.path.join(image2_dir, img2_name)

        score = dino_sim.calculate_similarity(img1_path, img2_path)
        scores.append(score)

    avg_score = np.mean(scores) if scores else 0.0
    return avg_score

def main():
    """
    示例用法
    """
    # 初始化DINO相似度计算器
    dino_sim = DINOSimilarity(model_name="facebook/dinov2-base")

    # 示例：计算两张图片的相似度
    image1_path = "/home/user/workspace/PycharmProject/similarity_score/generated/00006_00.jpg"
    image2_path = "/home/user/workspace/PycharmProject/similarity_score/original/00006_00.jpg"

    try:
        similarity_score = dino_sim.calculate_similarity(image1_path, image2_path)
        print(f"\n图片相似度: {similarity_score:.4f}")
        print(f"相似度百分比: {(similarity_score + 1) / 2 * 100:.2f}%")

        if similarity_score > 0.8:
            print("结论: 两张图片非常相似")
        elif similarity_score > 0.5:
            print("结论: 两张图片比较相似")
        elif similarity_score > 0.2:
            print("结论: 两张图片有一定相似性")
        else:
            print("结论: 两张图片差异较大")

    except FileNotFoundError as e:
        print(f"错误: 找不到图片文件 - {e}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()

