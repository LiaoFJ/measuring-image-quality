import torch
import gc
import argparse

import cleanfid.fid as fid
from fid_score.inception import InceptionV3

from similarity_score.cw_ssim import get_avg_cwssim_score, get_avg_cwssim_score_multi
from similarity_score.ms_ssim import get_avg_msssim_score
from similarity_score.dists import get_avg_dists_score
from similarity_score.dino_score import get_avg_dino_score
from lpips_score.lpips_calculate import lpips_score_dir
from ssim_score.calculate_ssim import get_ssim_score
from kid_score import get_kid_score
from cal_utils import save_similarity_results_to_log, save_sp_similarity_results_to_log

def calculate_lpips_score(real_dir, fake_dir):
    prediction = lpips_score_dir(real_dir=real_dir, fake_dir=fake_dir)
    return prediction

def calculate_ssim_score(real_dir, fake_dir):
    result_ssim_score = get_ssim_score(real_dir, fake_dir)
    return result_ssim_score

def calculate_kid_score(real_dir, fake_dir):
    return get_kid_score(real_dir, fake_dir)

def get_args():
    parser = argparse.ArgumentParser(description='Calculate image quality metrics')
    parser.add_argument('--real_dir', type=str, required=True,
                        help='Path to real/reference images directory')
    parser.add_argument('--fake_dir', type=str, required=True,
                        help='Path to fake/generated images directory')
    parser.add_argument('--unpaired', action='store_true',
                        help='calculate unpaired metrics or not')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                        help='CLIP model to use')
    parser.add_argument('--num-workers', type=int,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--real_flag', type=str, default='img',
                        help=('The modality of real path. '
                              'Default to img'))
    parser.add_argument('--fake_flag', type=str, default='img',
                        help=('The modality of real path. '
                              'Default to txt'))
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('--save-stats', action='store_true',
                        help=('Generate an npz archive from a directory of samples. '
                              'The first path is used as input and the second as output.'))
    parser.add_argument('--log-file', type=str, default='similarity_results.txt',
                        help='Path to save the results log file')
    args = parser.parse_args()
    return args

def main(real_test_dir, fake_test_dir, args):

    clip_fid_score = fid.compute_fid(real_test_dir, fake_test_dir, mode="clean", model_name="clip_vit_b_32")
    dino_score = get_avg_dino_score(real_test_dir, fake_test_dir)
    dists_score = get_avg_dists_score(real_test_dir, fake_test_dir)
    ms_ssim_score = get_avg_msssim_score(real_test_dir, fake_test_dir)
    cw_ssim_score = get_avg_cwssim_score(real_test_dir, fake_test_dir)


    result_lpips_score = calculate_lpips_score(real_dir=real_test_dir, fake_dir=fake_test_dir)
    result_ssim_score = calculate_ssim_score(real_dir=real_test_dir, fake_dir=fake_test_dir)
    fid_score = fid.compute_fid(real_test_dir, fake_test_dir)
    result_kid_score = calculate_kid_score(real_dir=real_test_dir, fake_dir=fake_test_dir)

    # 保存结果到字典
    results = {
        'clip_fid_score': clip_fid_score,
        'dino_score': dino_score,
        'cw_ssim_score': cw_ssim_score,
        'dists_score': dists_score,
        'ms_ssim_score': ms_ssim_score,

        'fid_score': fid_score,
        'lpips_score': result_lpips_score,
        'ssim_score': result_ssim_score,
        'kid_mean': result_kid_score[0],
        'kid_std': result_kid_score[1]
    }

    # 打印所有结果
    for key, value in results.items():
        print(f"{key}: {value}")

    # 保存到日志文件
    save_similarity_results_to_log(args, results, args.log_file)
    print(f"Results saved to: {args.log_file}")

def main_sp(real_test_dir, fake_test_dir, args):

    # clip_fid_score = fid.compute_fid(real_test_dir, fake_test_dir, mode="clean", model_name="clip_vit_b_32")
    # dino_score = get_avg_dino_score(real_test_dir, fake_test_dir)
    # dists_score = get_avg_dists_score(real_test_dir, fake_test_dir)
    # ms_ssim_score = get_avg_msssim_score(real_test_dir, fake_test_dir)
    cw_ssim_score = get_avg_cwssim_score_multi(
        real_test_dir,
        fake_test_dir,
        resize_to=(256, 256),  # 缩放到256x256，速度极快
        max_pairs=500,  # 最多计算500对
        max_workers=8  # 使用8个进程
    )

    # result_lpips_score = calculate_lpips_score(real_dir=real_test_dir, fake_dir=fake_test_dir)
    # result_ssim_score = calculate_ssim_score(real_dir=real_test_dir, fake_dir=fake_test_dir)
    # fid_score = fid.compute_fid(real_test_dir, fake_test_dir)
    # result_kid_score = calculate_kid_score(real_dir=real_test_dir, fake_dir=fake_test_dir)

    # 保存结果到字典
    results = {
        # 'clip_fid_score': clip_fid_score,
        # 'dino_score': dino_score,
        # 'dists_score': dists_score,
        # 'ms_ssim_score': ms_ssim_score,
        'cw_ssim_score': cw_ssim_score,
        # 'fid_score': fid_score,
        # 'lpips_score': result_lpips_score,
        # 'ssim_score': result_ssim_score,
        # 'kid_mean': result_kid_score[0],
        # 'kid_std': result_kid_score[1]
    }

    # 打印所有结果
    for key, value in results.items():
        print(f"{key}: {value}")

    # 保存到日志文件
    save_sp_similarity_results_to_log(args, results, args.log_file)
    print(f"Results saved to: {args.log_file}")


if __name__ == '__main__':
    args = get_args()
    # args.real_train_dir = "D:/workspace/dataset/Illustration_SR/trainB_SR_512"
    # real_test_dir = "/home/user/workspace/dataset/DressCode/upper_body/test_images"
    # fake_test_dir = "/home/user/workspace/generated_images/dresscode/paired/ours/paired"

    # main(args.real_dir, args.fake_dir, args)
    main_sp(args.real_dir, args.fake_dir, args)