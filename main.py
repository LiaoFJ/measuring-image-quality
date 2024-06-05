from clip_score.clip_score import calculate_clipscore
from clip_score.clip_aesthetic import calculate_clip_aesthetic
from fid_score.fid_calculate import fid_score
from lpips_score.lpips_calculate import lpips_score_dir


def test1():
    real_path = 'D:/workspace/dataset/illustration_SR_test/testB512'
    fake_path = "D:/workspace/generated_results/already/sd_lora_test"
    # fake_path = 'D:/workspace/dataset/illustration_SR_test/testB512'
    clip_score = calculate_clipscore(real_path=real_path, fake_path=fake_path)
    return


def test2():
    fake_path = "D:\\workspace\\generated_results\\already\\control_test"
    prediction = calculate_clip_aesthetic(fake_path)
    return prediction


def test3():
    real_path = "D:\workspace\dataset\Illustration_SR\\trainB_SR_512"
    fake_path = "D:\\workspace\\generated_results\\already\\control_test"
    fid_value = fid_score(real_path=real_path, fake_path=fake_path)
    return fid_value


def test4():
    real_dir = "D:\\workspace\\dataset\\illustration_SR_test\\testB512"
    fake_dir = "D:\\workspace\\generated_results\\already\\control_test"
    prediction = lpips_score_dir(real_dir=real_dir, fake_dir=fake_dir)
    return prediction


if __name__ == '__main__':
    test1()
