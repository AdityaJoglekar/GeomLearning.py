export CUDA_VISIBLE_DEVICES="0"
python -m bench \
    --dataset elasticity \
    --train true \
    --model_type -2 \
    --batch_size 4 \
    --epochs 500 \
    --learning_rate 1e-3 \
    --schedule OneCycleLR \
    --weight_decay 5e-5 \
    --one_cycle_div_factor 1e4 \
    --one_cycle_final_div_factor 1e4 \
    --one_cycle_pct_start 0.2 \
    --clip_grad_norm 1000.0 \
    --exp_name LNO_test \