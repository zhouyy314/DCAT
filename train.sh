#普通训练
CUDA_VISIBLE_DEVICES=5 python  pretrain.py \
    --batch_size 48 \
    --accum_iter 4 \
    --model mae_cat_small_256 \
    --mask_regular \
    --vis_mask_ratio 0.25 \
    --input_size 256 \
    --token_size 16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 10 \
    --blr 2e-4 \
    --weight_decay 0.05 \
    --pin_mem \
    --load_from ./pretrained_weights/checkpoint-79.pth \
    --data_path ./dataset/ \
    --log_dir ./work_dirs/fintune4cd/mae_cat_small_256 \
    --output_dir ./work_dirs/fintune4cd/mae_cat_small_256

#分布式训练 4个不同的变化检测数据集
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12534 pretrain.py \
    --batch_size 48 \
    --accum_iter 4 \
    --model mae_cat_small_256 \
    --mask_regular \
    --vis_mask_ratio 0.25 \
    --input_size 256 \
    --token_size 16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 10 \
    --blr 2e-4 \
    --weight_decay 0.05 \
    --pin_mem \
    --load_from ./pretrained_weights/checkpoint-399.pth \
    --data_path ./dataset/levir256 \
    --log_dir ./work_dirs/fintune_levir_std0.5/mae_cat_small_256 \
    --output_dir ./work_dirs/fintune_levir_std0.5/mae_cat_small_256