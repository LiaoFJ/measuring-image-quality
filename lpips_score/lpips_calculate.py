import lpips
import torch
import os
import torch.nn.functional as F

def get_base_filename(filename):
    """获取不带扩展名的文件名"""
    return os.path.splitext(filename)[0]

def resize_and_pad(img, target_height=1024, target_width=768):
    """将图像调整到目标尺寸，保持纵横比"""
    h, w = img.shape[-2:]

    # 计算缩放比例
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 先调整大小
    img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # 然后填充到目标尺寸
    pad_h = (target_height - new_h) // 2
    pad_w = (target_width - new_w) // 2
    pad_h_remainder = target_height - new_h - pad_h
    pad_w_remainder = target_width - new_w - pad_w

    img = F.pad(img, (pad_w, pad_w_remainder, pad_h, pad_h_remainder))
    return img

def main():
    loss_fn_vgg = lpips.LPIPS(net='alex')  # closer to "traditional" perceptual loss, when used for optimization

    img0 = torch.zeros(1, 3, 64, 64)  # image should be RGB, IMPORTANT: normalized to [-1,1]
    img1 = torch.zeros(1, 3, 64, 64)
    d = loss_fn_vgg(img0, img1)
    print(f"the lpips score is {d}")

def lpips_score_img(path0, path1):
    import argparse
    import lpips

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--version', type=str, default='0.1')

    opt = parser.parse_args()

    ## Initializing the model
    loss_fn = lpips.LPIPS(net='alex', version=opt.version)

    loss_fn.cuda()

    # Load images
    img0 = lpips.im2tensor(lpips.load_image(path0))  # RGB image from [-1,1]
    img1 = lpips.im2tensor(lpips.load_image(path1))

    # resize images
    img0 = resize_and_pad(img0)
    img1 = resize_and_pad(img1)

    img0 = img0.cuda()
    img1 = img1.cuda()

    # Compute distance
    dist01 = loss_fn.forward(img0, img1)
    print('Distance: %.3f' % dist01)


def lpips_score_dir(real_dir, fake_dir):

    ## Initializing the model
    loss_fn = lpips.LPIPS(net='alex')

    loss_fn.cuda()

    # 在循环前预先构建映射
    files = os.listdir(real_dir)
    fake_files = os.listdir(fake_dir)
    fake_file_map = {get_base_filename(f): f for f in fake_files}

    result = []

    for file in files:
        base_name = get_base_filename(file)

        if base_name in fake_file_map:
            fake_file = fake_file_map[base_name]
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(real_dir, file)))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(fake_dir, fake_file)))

            # resize images
            img0 = resize_and_pad(img0, 224)
            img1 = resize_and_pad(img1, 224)

            img0 = img0.cuda()
            img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0, img1)
            # 关键修复：移到CPU并立即清理GPU张量
            result.append(dist01.detach().cpu())

            # 清理GPU张量
            del img0, img1, dist01

    stacked_tensors = torch.stack(result)
    prediction = torch.mean(stacked_tensors)
    print('Avg: %.3f' % (prediction))

    return prediction

if __name__ == '__main__':
    path0 = "D:\\workspace\\PythonProject\\criteria\\test_img\\test_real.png"
    path1 = "D:\\workspace\\PythonProject\\criteria\\test_img\\test_gene.png"
    dir0 = "D:\\workspace\\dataset\\illustration_SR_test\\testB512"
    dir1 = "D:\\workspace\\generated_results\\already\\control_test"

    # lpips_score_img(path0=path0, path1=path1)
    prediction = lpips_score_dir(real_dir=dir0, fake_dir=dir1)
