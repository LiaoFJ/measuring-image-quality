from clip_score.clip_score import calculate_clipscore
from clip_score.clip_aesthetic import calculate_clip_aesthetic
from fid_score.fid_calculate import fid_score
from lpips_score.lpips_calculate import lpips_score_dir


def calculate_clip_score(real_dir, fake_dir):
    clip_score = calculate_clipscore(real_path=real_dir, fake_path=fake_dir)
    return clip_score

def clip_aesthetic(fake_dir):
    prediction = calculate_clip_aesthetic(fake_dir)
    return prediction


def calculate_fid64_score(real_train_dir, fake_dir):
    fid_value = fid_score(real_path=real_train_dir, fake_path=fake_dir)
    return fid_value

def calculate_lpips_score(real_dir, fake_dir):
    prediction = lpips_score_dir(real_dir=real_dir, fake_dir=fake_dir)
    return prediction


if __name__ == '__main__':
    real_train_dir = "D:/workspace/dataset/Illustration_SR/trainB_SR_512"
    real_test_dir = "D:/workspace/dataset/illustration_SR_test/testB512"
    # fake_test_dir = "D:/workspace/dataset/illustration_testB_rename"
    fake_test_dir = "D:/workspace/github/pytorch-CycleGAN-and-pix2pix/results/maps_cyclegan/test_latest/images"
    clip_score = calculate_clip_score(real_dir=real_test_dir, fake_dir=fake_test_dir)
    clip_aesthetic_score = clip_aesthetic(fake_dir=fake_test_dir)
    fid64 = calculate_fid64_score(real_train_dir=real_test_dir, fake_dir=fake_test_dir)
    lpips_score = calculate_lpips_score(real_dir=real_test_dir, fake_dir=fake_test_dir)
    print(f"Clip socre: {clip_score}, "f"Clip aesthetic: {clip_aesthetic_score} \n"
          f"Fid_64: {fid64}, Lpips score: {lpips_score}")
    # print(f"Fid_64: {fid64}, Lpips score: {lpips_score}")


