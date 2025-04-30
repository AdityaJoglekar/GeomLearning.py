export CUDA_VISIBLE_DEVICES="2"
python -m bench --dataset elasticity --eval True --schedule OneCycleLR \
    --epochs 500 --model_type 2 --learning_rate 1e-3 --weight_decay 0e-3 \
    --num_layers 5  --qk_norm false --batch_size 1 --exp_name elas_CA_TS2




# python -m bench \
#     --dataset elasticity \
#     --train true \
#     --model_type 3 \
#     --batch_size 1 \
#     --epochs 500 \
#     --learning_rate 1e-3 \
#     --schedule CosineAnnealingLR \
#     --weight_decay 0 \
#     --hidden_dim 128 \
#     --num_slices 64 \
#     --num_heads 8 \
#     --num_layers 8 \
#     --mlp_ratio 1 \
#     --exp_name CA_Elasticity_rel_error_print \
    # --supernodes 1024 
    

# export CUDA_VISIBLE_DEVICES="0"
# python -m bench \
#     --dataset elasticity \
#     --train true \
#     --model_type 0 \
#     --max_cases 10 \
#     --max_steps 100 \
#     --batch_size 1 \
#     --epochs 20 \
#     --learning_rate 1e-3 \
#     --schedule CosineAnnealingLR \
#     --weight_decay 1e-5 \
#     --hidden_dim 128 \
#     --num_slices 64 \
#     --num_heads 8 \
#     --num_layers 6 \
#     --mlp_ratio 1 \
#     --init_step 0 \
#     --exclude True \
#     --exp_name LNO_test \
#     # --supernodes 1024 