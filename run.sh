# For consistency distillation
python3 /tracto/DTG/consistency/dtg_cm_train.py \
--teacher_model_path /tracto/DTG/results/dtgsnapshot.pth.tar \
--output_dir /tracto/DTG/consistency_results \
--training_mode consistency_distillation \
--start_scales 40 \
--end_scales 10 \
--scale_mode progressive \
--total_training_steps 1000 \
--lr 1e-4

# For inference
python3 inference.py --snapshot /tracto/DTG/consistency_results/dtg_consistency_snapshot.pth.tar

# For training
python3 main.py