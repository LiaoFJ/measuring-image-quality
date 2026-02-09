import os

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from math import exp


def gaussian_kernel(window_size, sigma):
    """
    创建高斯核
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    创建用于SSIM计算的窗口
    """
    _1D_window = gaussian_kernel(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_single_scale(img1, img2, window, window_size, channel, size_average=True):
    """
    计算单尺度的SSIM
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class MSSSIM:
    """
    MS-SSIM (Multi-Scale Structural Similarity) 计算器
    """
    def __init__(self, window_size=11, channel=3, weights=None):
        """
        初始化MS-SSIM计算器

        Args:
            window_size: 窗口大小，默认11
            channel: 图像通道数，默认3 (RGB)
            weights: 各尺度的权重，默认None会使用标准权重
        """
        self.window_size = window_size
        self.channel = channel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 默认的5个尺度的权重（来自原始MS-SSIM论文）
        if weights is None:
            self.weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(self.device)
        else:
            self.weights = torch.FloatTensor(weights).to(self.device)

        self.window = create_window(window_size, channel).to(self.device)

        # 图像预处理转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def load_image(self, image_path):
        """
        加载图像并转换为tensor

        Args:
            image_path: 图像路径

        Returns:
            图像tensor
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)

    def calculate_msssim(self, img1, img2):
        """
        计算MS-SSIM

        Args:
            img1: 第一张图像tensor [B, C, H, W]
            img2: 第二张图像tensor [B, C, H, W]

        Returns:
            MS-SSIM分数
        """
        levels = self.weights.size(0)
        mssim = []
        mcs = []

        for i in range(levels):
            ssim_val = self._ssim(img1, img2)
            mssim.append(ssim_val)

            # 对比度比较
            cs = self._cs(img1, img2)
            mcs.append(cs)

            # 对图像进行下采样（除了最后一层）
            if i < levels - 1:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # 计算MS-SSIM
        # MS-SSIM = [l_M(x,y)]^αM · ∏(j=1 to M-1)[c_j(x,y)]^βj · [s_j(x,y)]^γj
        pow1 = mcs ** self.weights
        pow2 = mssim ** self.weights

        output = torch.prod(pow1[:-1] * pow2[-1])

        return output.item()

    def _ssim(self, img1, img2):
        """
        计算单尺度SSIM
        """
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def _cs(self, img1, img2):
        """
        计算对比度-结构比较
        """
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C2 = 0.03 ** 2

        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        return cs_map.mean()

    def compare_images(self, image_path1, image_path2):
        """
        比较两张图片的MS-SSIM

        Args:
            image_path1: 第一张图片路径
            image_path2: 第二张图片路径

        Returns:
            MS-SSIM分数 (0-1之间，越接近1表示越相似)
        """
        # 加载图片
        img1 = self.load_image(image_path1)
        img2 = self.load_image(image_path2)

        # 确保两张图片尺寸相同
        if img1.shape != img2.shape:
            # 调整img2的尺寸与img1相同
            img2 = F.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)

        # 计算MS-SSIM
        msssim_score = self.calculate_msssim(img1, img2)

        return msssim_score

def get_avg_msssim_score(image1_dir, image2_dir):
    """
    计算多对图片的平均MS-SSIM相似度分数

    Args:
        image_pairs: 包含图片路径对的列表 [(img1_path, img2_path), ...]

    Returns:
        平均DINO相似度分数 (float)
    """
    msssim_calculator = MSSSIM()
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

        score = msssim_calculator.compare_images(img1_path, img2_path)
        scores.append(score)

    avg_score = np.mean(scores) if scores else 0.0
    return avg_score



def main():
    """
    示例用法
    """
    print("初始化MS-SSIM计算器...")
    msssim_calculator = MSSSIM()

    # 示例：比较两张图片
    image1_path = "/home/user/workspace/PycharmProject/similarity_score/generated/00006_00.jpg"
    image2_path = "/home/user/workspace/PycharmProject/similarity_score/original/00006_00.jpg"

    try:
        print(f"\n正在计算图片相似度...")
        print(f"图片1: {image1_path}")
        print(f"图片2: {image2_path}")

        score = msssim_calculator.compare_images(image1_path, image2_path)

        print(f"\nMS-SSIM分数: {score:.6f}")
        print(f"相似度百分比: {score * 100:.2f}%")

        # 评价相似度
        if score > 0.95:
            print("结论: 图片几乎完全相同")
        elif score > 0.85:
            print("结论: 图片非常相似")
        elif score > 0.70:
            print("结论: 图片比较相似")
        elif score > 0.50:
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

