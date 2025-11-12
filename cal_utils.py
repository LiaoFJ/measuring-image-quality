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