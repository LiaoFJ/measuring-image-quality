source ~/miniconda3/etc/profile.d/conda.sh
conda activate measure  # 替换为你的环境名

# VITON dir
# /home/user/workspace/dataset/VTONHD/test/image

## VTION for catvton:
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/catvton/vitonhd-1024/paired
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/catvton/vitonhd-1024/unpaired --unpaired
#
## VTION for any2anytryon: /home/user/workspace/generated_images/vitonhd/paired/any2anytryon/paired
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/any2anytryon/paired
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/any2anytryon/unpaired --unpaired
#
## VTION for ootd
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/ootd/viton/paired
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/ootd/viton/unpaired --unpaired
#
## VTION for HRVTION
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/HRVITON/paired
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/HRVITON/unpaired --unpaired
#
## VTION for StableVITON
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/StableVITON/paired
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/StableVITON/unpaired --unpaired

## VTION for idm_vton:
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/idmvton/paired
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/idmvton/unpaired --unpaired

## VTION for LayerDiffusion:
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/ours/0929paired_result
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/ours/unpaired/unpaired_05Nov2025_20000 --unpaired

#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/ours/paired/11Nov2025_23500_paired
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/ours/unpaired/11Nov2025_23500_unpaired --unpaired


#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/ablation/vton_vanilla_fill_redux --unpaired
#python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/ablation/ablation_fill_incontext_resized_single --unpaired
python calculate_score.py --real_dir /home/user/workspace/dataset/VTONHD/test/image --fake_dir /home/user/workspace/generated_images/vitonhd/paired/ablation/ft_redux --unpaired


# ==================== dress code =============================

## Dresscode for catvton:
#python calculate_score.py --real_dir /home/user/workspace/dataset/DressCode/upper_body/test_images --fake_dir /home/user/workspace/generated_images/dresscode/paired/catvton/dresscode-1024/paired/upper_body
#python calculate_score.py --real_dir /home/user/workspace/dataset/DressCode/upper_body/test_images --fake_dir /home/user/workspace/generated_images/dresscode/paired/catvton/dresscode-1024/unpaired/upper_body --unpaired
#
# Dresscode for any2anytryon: /home/user/workspace/generated_images/vitonhd/paired/any2anytryon/paired
#python calculate_score.py --real_dir /home/user/workspace/dataset/DressCode/upper_body/test_images --fake_dir /home/user/workspace/generated_images/dresscode/paired/any2anytryon/paired
#python calculate_score.py --real_dir /home/user/workspace/dataset/DressCode/upper_body/test_images --fake_dir /home/user/workspace/generated_images/dresscode/paired/any2anytryon/unpaired --unpaired

## Dresscode for ootd
#python calculate_score.py --real_dir /home/user/workspace/dataset/DressCode/upper_body/test_images --fake_dir /home/user/workspace/generated_images/dresscode/paired/ootd/dresscode/paired/upper_body
#python calculate_score.py --real_dir /home/user/workspace/dataset/DressCode/upper_body/test_images --fake_dir /home/user/workspace/generated_images/dresscode/paired/ootd/dresscode/unpaired/upper_body --unpaired

## Dresscode for LayerDiffusion:
#python calculate_score.py --real_dir /home/user/workspace/dataset/DressCode/upper_body/test_images --fake_dir /home/user/workspace/generated_images/dresscode/paired/ours/paired_3Nov2025 --log-file score_results_Dresscode.txt
