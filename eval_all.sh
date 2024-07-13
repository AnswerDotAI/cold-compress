python parallelize_evals.py \
	--config_names random l2 scissor window \
	--tasks truthfulqa rulerqa rulerniah rulervt rulercwe scrollsquality musique squality dolomites qmsum reprobench \
	--cache_sizes 0.75 0.5 0.25 0.1 0.05 \
	--num_samples 500 \
	--add_full \
	--checkpoint_path checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/model.pth

python parallelize_evals.py \
	--config_names random l2 scissor window \
	--tasks truthfulqa rulerqa rulerniah rulervt rulercwe scrollsquality musique squality dolomites qmsum reprobench \
	--cache_sizes 0.75 0.5 0.25 0.1 0.05 \
	--num_samples 500 \
	--add_full \
	--checkpoint_path checkpoints/Qwen/Qwen2-7B-Instruct/model.pth
