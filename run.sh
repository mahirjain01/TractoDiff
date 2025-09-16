# For consistency distillation
python3 /med/TractoDiff/consistency/dtg_cm_train.py \
--teacher_model_path /med/TractoDiff/snapshots/dtgsnapshot.pth.tar \
--output_dir /med/TractoDiff/consistency_results \
--training_mode consistency_distillation \
--start_scales 40 \
--end_scales 10 \
--scale_mode progressive \
--total_training_steps 1000 \
--lr 1e-4

# For inference
python3 inference.py --snapshot /med/TractoDiff/snapshots/dtg_consistency_snapshot.pth.tar

# For training
python3 main.py

conda activate /med/TractoDiff/environment && CUDA_LAUNCH_BLOCKING=1 python src/generate_streamline.py --subject sub-1119 --bundle AF_L --dataset_file /med/TractoDiff/data/testset/sub-1119/sub-1119.hdf5 --wm_loc /med/TractoDiff/data/testset/sub-1119/sub-1119-generated_approximated_mask.nii.gz --model_path /med/TractoDiff/output_dir/models/TractoDiff_6.pth --seed_trk /med/TractoDiff/data/testset/sub-1119/tractography/sub-1119__AF_L.trk --output_trk /med/TractoDiff/visualizations/generated.trk


conda activate ./environment