python parallelize_evals.py \
	--config_names random l2 scissor window \
	--tasks truthfulqa rulerqa rulerniah rulervt rulercwe scrollsquality dolomites musique squality qmsum \
	--cache_sizes 8192 4096 2048 1024 512 256 128 \
	--num_samples 100 \
	--checkpoint_path ./checkpoints/Qwen/Qwen2-1.5B-Instruct/model.pth
