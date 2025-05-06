# For consistency distillation
python3 /tracto/TractoDiff/consistency/dtg_cm_train.py \
--teacher_model_path /tracto/TractoDiff/snapshots/dtgsnapshot.pth.tar \
--output_dir /tracto/TractoDiff/consistency_results \
--training_mode consistency_distillation \
--start_scales 40 \
--end_scales 10 \
--scale_mode progressive \
--total_training_steps 1000 \
--lr 1e-4

# For inference
python3 inference.py --snapshot /tracto/TractoDiff/snapshots/dtg_consistency_snapshot.pth.tar

# For training
python3 main.py

conda activate /tracto/TractoDiff/environment && CUDA_LAUNCH_BLOCKING=1 python src/generate_streamline.py --subject sub-1119 --bundle AF_L --dataset_file /tracto/TractoDiff/data/testset/sub-1119/sub-1119.hdf5 --wm_loc /tracto/TractoDiff/data/testset/sub-1119/sub-1119-generated_approximated_mask.nii.gz --model_path /tracto/TractoDiff/output_dir/models/TractoDiff_6.pth --seed_trk /tracto/TractoDiff/data/testset/sub-1119/tractography/sub-1119__AF_L.trk --output_trk /tracto/TractoDiff/visualizations/generated.trk


