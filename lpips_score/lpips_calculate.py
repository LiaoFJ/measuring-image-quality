import lpips
import torch
import os

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

    img0 = img0.cuda()
    img1 = img1.cuda()

    # Compute distance
    dist01 = loss_fn.forward(img0, img1)
    print('Distance: %.3f' % dist01)


def lpips_score_dir(real_dir, fake_dir):

    ## Initializing the model
    loss_fn = lpips.LPIPS(net='alex')

    loss_fn.cuda()

    files = os.listdir(real_dir)
    result = []

    for file in files:
        fake_file = file.replace("draw", "photo")
        if (os.path.exists(os.path.join(fake_dir, fake_file))):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(real_dir, file)))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(fake_dir, fake_file)))

            img0 = img0.cuda()
            img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0, img1)
            print('%s: %.3f' % (file, dist01))
            result.append(dist01)
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
