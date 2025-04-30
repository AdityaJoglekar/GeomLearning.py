export CUDA_VISIBLE_DEVICES="0"
python -m bench \
    --dataset cylinder_flow \
    --train true \
    --model_type 0 \
    --max_cases 1000 \
    --max_steps 600 \
    --batch_size 1 \
    --epochs 20 \
    --learning_rate 1e-3 \
    --schedule OneCycleLR \
    --weight_decay 1e-5 \
    --hidden_dim 128 \
    --num_slices 64 \
    --num_heads 8 \
    --num_layers 6 \
    --mlp_ratio 1 \
    --init_step 0 \
    --exclude False \
    --exp_name OG_1000train_100test_600s_20ep

python -m bench \
    --dataset cylinder_flow \
    --train true \
    --model_type 3 \
    --max_cases 1000 \
    --max_steps 600 \
    --batch_size 1 \
    --epochs 20 \
    --learning_rate 1e-3 \
    --schedule OneCycleLR \
    --weight_decay 1e-5 \
    --hidden_dim 128 \
    --num_slices 64 \
    --num_heads 8 \
    --num_layers 6 \
    --mlp_ratio 1 \
    --init_step 0 \
    --exclude False\
    --exp_name Qcond_slicingscaleliketransolver_1000train_100test_600s_20ep