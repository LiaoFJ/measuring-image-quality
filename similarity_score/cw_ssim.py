import os
import random
import concurrent.futures
from functools import partial
from PIL import Image
from ssim import SSIM

# 这是一个辅助函数，用于在单独的进程中计算一对图像的CW-SSIM
# 它必须定义在顶层，以便被其他进程调用
def _calculate_cw_ssim_for_pair(img_paths, resize_to=None):
    """
    计算单个图像对的CW-SSIM分数。
    """
    img1_path, img2_path = img_paths
    try:
        image1 = Image.open(img1_path)
        image2 = Image.open(img2_path)

        # 如果指定了resize_to，则在计算前缩放图像以提速
        if resize_to:
            # 使用LANCZOS以获得高质量的缩放效果
            image1 = image1.resize(resize_to, Image.LANCZOS)
            image2 = image2.resize(resize_to, Image.LANCZOS)

        score = SSIM(image1).cw_ssim_value(image2)
        return score
    except Exception as e:
        print(f"Warning: Failed to process pair ({os.path.basename(img1_path)}, {os.path.basename(img2_path)}): {e}")
        return None  # 对于处理失败的图像对返回None


def get_avg_cwssim_score_multi(
        image1_dir,
        image2_dir,
        resize_to=(256, 256),  # 默认缩放以大幅提速，设为None则使用原图尺寸
        max_pairs=None,  # 限制最大计算对数，None表示计算全部
        max_workers=None  # 并行进程数，None表示使用所有CPU核心
):
    """
    使用多进程并行计算两个目录中对应图像的平均CW-SSIM分数。

    Args:
        image1_dir: 第一个图像目录。
        image2_dir: 第二个图像目录。
        resize_to (tuple, optional): 将图像缩放到指定尺寸 (width, height) 以加速。默认为 (256, 256)。
        max_pairs (int, optional): 随机抽样计算的最大图像对数量。默认为 None (全部计算)。
        max_workers (int, optional): 使用的并行工作进程数。默认为 None (自动决定)。

    Returns:
        float: 平均CW-SSIM分数。
    """
    # 获取并排序图像文件
    image_files1 = sorted([f for f in os.listdir(image1_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    image_files2 = sorted([f for f in os.listdir(image2_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    if len(image_files1) != len(image_files2):
        raise ValueError(f"目录中的图像数量不匹配: {len(image_files1)} vs {len(image_files2)}")

    # 创建完整的图像路径对
    image_pairs = [(os.path.join(image1_dir, f1), os.path.join(image2_dir, f2)) for f1, f2 in
                   zip(image_files1, image_files2)]
    num_total_pairs = len(image_pairs)

    # 如果设置了max_pairs，则进行随机抽样
    if max_pairs is not None and max_pairs < num_total_pairs:
        print(f"Randomly sampling {max_pairs} out of {num_total_pairs} pairs...")
        image_pairs = random.sample(image_pairs, max_pairs)

    if not image_pairs:
        raise ValueError("No image pairs to process.")

    print(f"Processing {len(image_pairs)} image pairs using up to {max_workers or os.cpu_count()} workers...")

    score_list = []
    # 使用functools.partial来固定worker函数的resize_to参数
    worker_func = partial(_calculate_cw_ssim_for_pair, resize_to=resize_to)

    # 使用ProcessPoolExecutor进行并行计算
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # map方法会按顺序返回结果
        results = executor.map(worker_func, image_pairs)

        # 收集所有成功计算的分数
        for score in results:
            if score is not None:
                score_list.append(score)

    if not score_list:
        raise ValueError("No valid scores were computed. All pairs failed.")

    return sum(score_list) / len(score_list)

def get_avg_cwssim_score(image1_dir, image2_dir):
    """
    计算两个目录中对应图像的平均CW-SSIM分数

    Args:
        image1_dir: 第一个图像目录
        image2_dir: 第二个图像目录

    Returns:
        平均CW-SSIM分数
    """
    import os

    score_list = []

    # 获取并排序图像文件
    image_files1 = sorted([f for f in os.listdir(image1_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    image_files2 = sorted([f for f in os.listdir(image2_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    if len(image_files1) != len(image_files2):
        raise ValueError(f"目录中的图像数量不匹配: {len(image_files1)} vs {len(image_files2)}")

    print(f"Processing {len(image_files1)} image pairs...")

    for i, (img1_name, img2_name) in enumerate(zip(image_files1, image_files2)):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(image_files1)}")

        img1_path = os.path.join(image1_dir, img1_name)
        img2_path = os.path.join(image2_dir, img2_name)

        try:
            image1 = Image.open(img1_path)
            image2 = Image.open(img2_path)
            score = SSIM(image1).cw_ssim_value(image2)
            score_list.append(score)
        except Exception as e:
            print(f"Warning: Failed to process {img1_name}: {e}")
            continue

    if not score_list:
        raise ValueError("No valid scores computed")

    return sum(score_list) / len(score_list)

def main():

    # 示例图片路径
    image1_path = "/home/user/workspace/PycharmProject/similarity_score/generated/00006_00.jpg"
    image2_path = "/home/user/workspace/PycharmProject/similarity_score/original/00006_00.jpg"

    try:
        print("初始化简化版CW-SSIM计算器...")

        print(f"\n正在计算图片相似度...")
        print(f"图片1: {image1_path}")
        print(f"图片2: {image2_path}")

        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)
        score = SSIM(image1).cw_ssim_value(image2)

        print(f"\nCW-SSIM分数 (简化版): {score:.6f}")
        print(f"相似度百分比: {score * 100:.2f}%")

    except FileNotFoundError as e:
        print(f"错误: 找不到图片文件 - {e}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

def main_multi():
    # 示例：计算两个目录的平均分
    # 请将下面的路径替换为你的实际目录
    real_dir = "/home/user/workspace/dataset/clean_viton/cloth"
    fake_dir = "/home/user/workspace/generated_images/tryoff/any2any_vton_384"

    try:
        print("开始计算两个目录的平均CW-SSIM分数...")
        # 调用并行版本函数，可以调整参数以平衡速度和精度
        avg_score = get_avg_cwssim_score_multi(
            real_dir,
            fake_dir,
            resize_to=(256, 256),  # 缩放到256x256，速度极快
            max_pairs=500,         # 最多计算500对
            max_workers=8          # 使用8个进程
        )
        print(f"\n计算完成。")
        print(f"平均 CW-SSIM 分数: {avg_score:.6f}")

    except FileNotFoundError as e:
        print(f"错误: 找不到目录 - {e}")
    except Exception as e:
        print(f"计算过程中发生错误: {e}")

if __name__ == "__main__":
    # main()
    main_multi()

