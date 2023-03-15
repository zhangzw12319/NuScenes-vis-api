cd ..

# v1.0-mini
# python ./visualize_panoptic.py \
# --dataset D:/nuscenes/ \
# --version v1.0-mini \
# --split val \
# --pkl_path --D:/nuscenes_pkl/mini \
# --predictions path_to_npz_files_folder \
# --do_instances \
# --gt_classwise \
# --pred_classwise \
# --render_lidar \
# --dark_mode # optional

# v1.0-trainval
python ./visualize_panoptic.py \
--dataset D:/nuscenes/ \
--version v1.0-trainval \
--split val \
--pkl_path D:/nuscenes_pkl/trainval \
--do_instances \
--gt_classwise \
--pred_classwise \
--render_lidar \

# v1.0-test
# python ./visualize_panoptic.py \
# --dataset D:/nuscenes/ \
# --version v1.0-test \
# --pkl_path D:/nuscenes_pkl/test \
# --do_instances \
# --pred_classwise \
# --render_lidar \