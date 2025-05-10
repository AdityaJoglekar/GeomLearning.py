export CUDA_VISIBLE_DEVICES="0"
python -m bench \
    --dataset elasticity \
    --train true \
    --model_type -4 \
    --batch_size 4 \
    --epochs 500 \
    --learning_rate 1e-3 \
    --schedule OneCycleLR \
    --weight_decay 5e-5 \
    --one_cycle_div_factor 25 \
    --one_cycle_final_div_factor 1e4 \
    --one_cycle_pct_start 0.05 \
    --clip_grad_norm 1.0 \
    --exp_name LNO_unet_test_nlno3_nlayer1_postln_ln_ndim128_nmode128_linearattnproj_OCLR251e4005 \