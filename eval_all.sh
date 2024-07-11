python parallelize_evals.py \
	--config_names random l2 scissor window \
	--tasks truthfulqa rulerqa rulerniah rulervt rulercwe scrollsquality musique squality dolomites qmsum reprobench \
	--cache_sizes 0.75 0.5 0.25 0.1 0.05 \
	--num_samples 100 \
	--checkpoint_path ./checkpoints/Qwen/Qwen2-1.5B-Instruct/model.pth

python parallelize_evals.py \
	--config_names full \
	--tasks truthfulqa rulerqa rulerniah rulervt rulercwe scrollsquality musique squality dolomites qmsum reprobench \
	--cache_sizes 1.0 \
	--num_samples 100 \
	--checkpoint_path ./checkpoints/Qwen/Qwen2-1.5B-Instruct/model.pth
