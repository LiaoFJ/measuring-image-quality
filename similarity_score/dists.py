import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from DISTS_pytorch import DISTS


class DISTSCalculator:
    """
    DISTS (Deep Image Structure and Texture Similarity) 计算器
    """
    def __init__(self):
        """
        初始化DISTS计算器
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化DISTS模型
        self.model = DISTS().to(self.device)
        self.model.eval()

        # 图像预处理转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为tensor，范围[0, 1]
        ])

    def load_image(self, image_path):
        """
        加载图像并转换为tensor

        Args:
            image_path: 图像路径

        Returns:
            图像tensor [1, C, H, W]，范围[0, 1]
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # 添加batch维度
        return image_tensor.to(self.device)

    def calculate_dists(self, X, Y, require_grad=False, batch_average=True):
        """
        计算DISTS分数

        Args:
            X: 第一批图像tensor (N, C, H, W)，范围[0, 1]
            Y: 第二批图像tensor (N, C, H, W)，范围[0, 1]
            require_grad: 是否需要梯度（用于训练）
            batch_average: 是否对batch取平均

        Returns:
            DISTS分数（越小表示越相似）
        """
        with torch.set_grad_enabled(require_grad):
            dists_value = self.model(X, Y, require_grad=require_grad, batch_average=batch_average)

        return dists_value

    def compare_images(self, image_path1, image_path2):
        """
        比较两张图片的DISTS

        Args:
            image_path1: 第一张图片路径
            image_path2: 第二张图片路径

        Returns:
            DISTS分数（越小表示越相似，0表示完全相同）
        """
        # 加载图片
        X = self.load_image(image_path1)
        Y = self.load_image(image_path2)

        # 确保两张图片尺寸相同
        if X.shape != Y.shape:
            Y = F.interpolate(Y, size=X.shape[2:], mode='bilinear', align_corners=False)

        # 计算DISTS
        with torch.no_grad():
            dists_score = self.model(X, Y, require_grad=False, batch_average=True)

        return dists_score.item()

    def compare_batch_images(self, image_paths1, image_paths2):
        """
        批量比较多对图片的DISTS

        Args:
            image_paths1: 第一组图片路径列表
            image_paths2: 第二组图片路径列表

        Returns:
            DISTS分数列表
        """
        assert len(image_paths1) == len(image_paths2), "两组图片数量必须相同"

        # 加载所有图片
        X_list = [self.load_image(path) for path in image_paths1]
        Y_list = [self.load_image(path) for path in image_paths2]

        # 合并成batch
        X = torch.cat(X_list, dim=0)  # (N, C, H, W)
        Y = torch.cat(Y_list, dim=0)  # (N, C, H, W)

        # 计算DISTS
        with torch.no_grad():
            dists_scores = self.model(X, Y, require_grad=False, batch_average=False)

        return dists_scores.cpu().numpy().tolist()

def get_avg_dists_score(image1_dir, image2_dir):
    """
    计算多对图片的平均DINO相似度分数

    Args:
        image_pairs: 包含图片路径对的列表 [(img1_path, img2_path), ...]

    Returns:
        平均DINO相似度分数 (float)
    """
    dists_calculator = DISTSCalculator()

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

        score = dists_calculator.compare_images(img1_path, img2_path)
        scores.append(score)

    avg_score = np.mean(scores) if scores else 0.0
    return avg_score


def main():
    """
    示例用法
    """
    print("=" * 60)
    print("DISTS (Deep Image Structure and Texture Similarity) 测试")
    print("=" * 60)

    # 初始化DISTS计算器
    print("\n初始化DISTS计算器...")
    dists_calculator = DISTSCalculator()

    # 示例1：比较两张图片
    print("\n" + "-" * 60)
    print("[示例1] 单对图片比较:")
    print("-" * 60)

    image1_path = "/home/user/workspace/PycharmProject/similarity_score/generated/00006_00.jpg"
    image2_path = "/home/user/workspace/PycharmProject/similarity_score/original/00006_00.jpg"

    try:
        print(f"\n正在计算图片相似度...")
        print(f"图片1: {image1_path}")
        print(f"图片2: {image2_path}")

        score = dists_calculator.compare_images(image1_path, image2_path)

        print(f"\nDISTS分数: {score:.6f}")
        print("注意: DISTS分数越小表示越相似，0表示完全相同")

        # 评价相似度
        if score < 0.05:
            print("结论: 图片几乎完全相同")
        elif score < 0.1:
            print("结论: 图片非常相似")
        elif score < 0.2:
            print("结论: 图片比较相似")
        elif score < 0.3:
            print("结论: 图片有一定相似性")
        else:
            print("结论: 图片差异较大")

    except FileNotFoundError as e:
        print(f"错误: 找不到图片文件 - {e}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()
