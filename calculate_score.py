import torch
import gc
import argparse

from clip_score.clip_score import calculate_clipscore
from clip_score.clip_aesthetic import calculate_clip_aesthetic
from fid_score.fid_calculate import fid_score
from fid_score.inception import InceptionV3
from lpips_score.lpips_calculate import lpips_score_dir
from ssim_score.calculate_ssim import get_ssim_score
from kid_score import get_kid_score

from cal_utils import save_results_to_log


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
    parser.add_argument('--log-file', type=str, default='score_results.txt',
                        help='Path to save the results log file')
    args = parser.parse_args()
    return args

def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def calculate_clip_score(real_dir, fake_dir, args):
    result_clip_score = calculate_clipscore(real_path=real_dir, fake_path=fake_dir, args=args)
    return result_clip_score

def clip_aesthetic(fake_dir):
    prediction = calculate_clip_aesthetic(fake_dir)
    return prediction


def calculate_fid_score(real_dir, fake_dir, args):
    fid_value = fid_score(real_path=real_dir, fake_path=fake_dir, args=args)
    return fid_value

def calculate_lpips_score(real_dir, fake_dir):
    prediction = lpips_score_dir(real_dir=real_dir, fake_dir=fake_dir)
    return prediction

def calculate_ssim_score(real_dir, fake_dir):
    result_ssim_score = get_ssim_score(real_dir, fake_dir)
    return result_ssim_score

def calculate_kid_score(real_dir, fake_dir):
    return get_kid_score(real_dir, fake_dir)


def main(real_test_dir, fake_test_dir, args):

    if not args.unpaired:
        print(f"args.unpaired: {args.unpaired:}")
        result_lpips_score = calculate_lpips_score(real_dir=real_test_dir, fake_dir=fake_test_dir)
        result_ssim_score = calculate_ssim_score(real_dir=real_test_dir, fake_dir=fake_test_dir)
        result_clip_score = calculate_clip_score(real_dir=real_test_dir, fake_dir=fake_test_dir, args=args)

        clear_gpu_cache()
    else:
        result_lpips_score = None
        result_ssim_score = None
        result_clip_score = None

    clip_aesthetic_score = clip_aesthetic(fake_dir=fake_test_dir)
    fid = calculate_fid_score(real_dir=real_test_dir, fake_dir=fake_test_dir,args=args)
    result_kid_score = calculate_kid_score(real_dir=real_test_dir, fake_dir=fake_test_dir)


    print(f"Clip socre: {result_clip_score}, "f"Clip aesthetic: {clip_aesthetic_score} \n"
          f"Fid_: {fid}, Lpips score: {result_lpips_score}, SSIM score: {result_ssim_score}, \n"
          f"KID_mean: {result_kid_score[0]}, KID_std: {result_kid_score[1]}")

    # 保存结果到字典
    results = {
        'clip_score': result_clip_score,
        'clip_aesthetic': clip_aesthetic_score,
        'fid': fid,
        'lpips_score': result_lpips_score,
        'ssim_score': result_ssim_score,
        'kid_mean': result_kid_score[0],
        'kid_std': result_kid_score[1]
    }

    # 保存到日志文件
    save_results_to_log(args, results, args.log_file)
    print(f"Results saved to: {args.log_file}")

if __name__ == '__main__':
    args = get_args()
    # args.real_train_dir = "D:/workspace/dataset/Illustration_SR/trainB_SR_512"
    # real_test_dir = "/home/user/workspace/dataset/DressCode/upper_body/test_images"
    # fake_test_dir = "/home/user/workspace/generated_images/dresscode/paired/ours/paired"
    main(args.real_dir, args.fake_dir, args)