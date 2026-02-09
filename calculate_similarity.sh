source ~/miniconda3/etc/profile.d/conda.sh
conda activate measure  # 替换为你的环境名

# any2any tryon
#python calculate_similarity.py --real_dir /home/user/workspace/dataset/clean_viton/cloth --fake_dir /home/user/workspace/generated_images/tryoff/any2any_vton_384
python calculate_similarity.py --real_dir /home/user/workspace/dataset/clean_viton/cloth --fake_dir /home/user/workspace/generated_images/tryoff/tryoffdiff/output_vton
#python calculate_similarity.py --real_dir /home/user/workspace/dataset/clean_viton/cloth --fake_dir /home/user/workspace/generated_images/tryoff/ours/tvton-ver9_lora_redux_vton_same_ids_11Nov2025_2_off_cropped

