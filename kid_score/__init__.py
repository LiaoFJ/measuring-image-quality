from torch_fidelity import calculate_metrics

def get_kid_score(original_dir, generated_dir):
    metrics = calculate_metrics(
        input1=original_dir,  # 真实图像目录
        input2=generated_dir,  # 生成图像目录
        kid=True,
        gpu=0
    )
    print(f"KID: {metrics['kernel_inception_distance_mean']:.6f} ± {metrics['kernel_inception_distance_std']:.6f}")
    return metrics['kernel_inception_distance_mean'], metrics['kernel_inception_distance_std']

if __name__ == '__main__':
    original_dir='/home/user/workspace/dataset/VTON-HD/test/image'
    generated_dir='/home/user/workspace/generated_images/vitonhd/paired/catvton/vitonhd-1024/paired'
    get_kid_score(original_dir, generated_dir)