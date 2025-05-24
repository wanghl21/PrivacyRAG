export CUDA_VISIBLE_DEVICES=3
data_path=BioASQ-TaskB
pretrained_model_path=/data/huangyuanhong/shandong/xRAG/pretrained_model/finetune/qwen/bge-large/last/model
retriever_path=BAAI/bge-large-en-v1.5

### Utility Evaluation
python -m src.eval.run_eval \
    --data $data_path \
    --model_name_or_path $pretrained_model_path \
    --retriever_name_or_path $retriever_path

### Target Adaptive Attack
python -m src.eval.run_eval \
    --data $data_path \
    --model_name_or_path $pretrained_model_path \
    --retriever_name_or_path $retriever_path \
    --target_adaptive_attack True

### Untarget Inversion Attack
python -m src.eval.run_eval \
    --data $data_path \
    --model_name_or_path $pretrained_model_path \
    --retriever_name_or_path $retriever_path \
    --untarget_inversion_attack True