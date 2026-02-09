import datetime

def save_results_to_log(args, results, log_file):
    """保存参数和结果到日志文件"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 80 + "\n")
        f.write("Arguments:\n")
        f.write(f"  Real Directory: {args.real_dir}\n")
        f.write(f"  Fake Directory: {args.fake_dir}\n")
        f.write("-" * 80 + "\n")
        f.write("Results:\n")
        f.write(f"  CLIP Score: {results['clip_score']}\n")
        f.write(f"  CLIP Aesthetic: {results['clip_aesthetic']}\n")
        f.write(f"  FID: {results['fid']}\n")
        f.write(f"  LPIPS Score: {results['lpips_score']}\n")
        f.write(f"  SSIM Score: {results['ssim_score']}\n")
        f.write(f"  KID Mean: {results['kid_mean']}\n")
        f.write(f"  KID Std: {results['kid_std']}\n")
        f.write("=" * 80 + "\n\n")

def save_similarity_results_to_log(args, results, log_file):
    """保存参数和结果到日志文件"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 80 + "\n")
        f.write("Arguments:\n")
        f.write(f"  Real Directory: {args.real_dir}\n")
        f.write(f"  Fake Directory: {args.fake_dir}\n")
        f.write("-" * 80 + "\n")
        f.write("Results:\n")
        f.write(f"  clip_fid_score: {results['clip_fid_score']}\n")
        f.write(f"  dino_score: {results['dino_score']}\n")
        f.write(f"  sw_ssim_score: {results['sw_ssim_score']}\n")
        f.write(f"  dists_score: {results['dists_score']}\n")
        f.write(f"  ms_ssim_score: {results['ms_ssim_score']}\n")

        f.write(f"  fid_score: {results['fid_score']}\n")
        f.write(f"  lpips_score: {results['lpips_score']}\n")
        f.write(f"  ssim_score: {results['ssim_score']}\n")
        f.write(f"  KID Mean: {results['kid_mean']}\n")
        f.write(f"  KID Std: {results['kid_std']}\n")
        f.write("=" * 80 + "\n\n")

def save_sp_similarity_results_to_log(args, results, log_file):
    """保存参数和结果到日志文件"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 80 + "\n")
        f.write("Arguments:\n")
        f.write(f"  Real Directory: {args.real_dir}\n")
        f.write(f"  Fake Directory: {args.fake_dir}\n")
        f.write("-" * 80 + "\n")
        f.write("Results:\n")
        f.write(f"  cw_ssim_score: {results['cw_ssim_score']}\n")
        f.write("=" * 80 + "\n\n")
