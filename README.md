# Image quality criteria

This repository provides a batch-wise quick processing for calculating image quality scores including CLIP scores, FID scores LPIPS scores and CLIP aesthetic
scores. The project structure is adapted from [clip aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor), [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [pytorch-fid](https://github.com/leognha/PyTorch-FID-score) and [CLIP](https://github.com/openai/CLIP) and referenced from the [clip-score](https://github.com/Taited/clip-score). Please go to the original repo for more reference.
## Installation

Requirements:
- Install PyTorch:
  ```
  pip install torch  
  ```
- Install CLIP:
  ```
  pip install git+https://github.com/openai/CLIP.git
  ```
- other requirements
  ```
  pip install -r requirements.txt
  ```

### Image Files

This project is designed to process real images and generated images. And all images should be stored in a single directory. The image files can be in either `.png` or `.jpg` format.

### File Number and Naming
- for CLIP score: 

  The number of files in the real image directory should be exactly equal to the number of files in the fake image directory. Additionally, the files in the real image directory and fake image directory should be paired by file name. For instance, if there is a `cat.png` in the real image directory, there should be a corresponding `cat.png` in the fake image directory.

- for LPIPS:

  same as clip score. 

- for FID score

  there is not much limitation for the real image data dir and fake image data dir but ensure the path.
  
- for CLIP aesthetic score:
  
  just need to make sure the fake path dir.
  
### Directory Structure Example

Below is an example of the expected directory structure (for CLIP score and LPIPS score):

```
├── path/to/real_image
│   ├── cat.png
│   ├── dog.png
│   └── bird.jpg
└── path/to/fake_image
    ├── cat.png
    ├── dog.png
    └── bird.jpg
```

For FID score:
```
├── path/to/real_image
│   ├── ...
│   └── ...
└── path/to/fake_image
    ├── ...
    └── ...
```
For CLIP aesthetic score:
```
└── path/to/fake_image
    ├── ...
    └── ...
```
## Usage

To compute the criteria Run the following command:

```
python calculate_socre.py
```


## License

This implementation is licensed under the MIT License.

The project structure is adapted from [mseitzer's pytorch-fid](https://github.com/mseitzer/pytorch-fid) project. The CLIP model is adapted from [OpenAI's CLIP](https://github.com/openai/CLIP).

The CLIP Score was introduced in OpenAI's [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).
