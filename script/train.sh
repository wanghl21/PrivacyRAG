## pretrain
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    -m \
    src.training.train \
    --config config/pretrain.yaml

## finetune
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    -m \
    src.training.train \
    --config config/finetune.yaml