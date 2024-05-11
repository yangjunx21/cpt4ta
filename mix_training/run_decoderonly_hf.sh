# data_name=advbench_30_ultrafeedback_30_alignsft_GA/vicuna_format
train_data_dir=/data/yangjunxiao/ai4ta/continue_pretraining/data/aml_training_data.json
val_data_dir=/data/yangjunxiao/ai4ta/continue_pretraining/data/aml_training_data.json

root_dir=/data/yangjunxiao/ai4ta/continue_pretraining/mix_training
root_dir=.

model_name=/data/yangjunxiao/huggingface_pretrained_models/ChatGLM3/model/chatglm3-6b-base

loss_type=GD
max_epoch=2
max_length=2048
deepspeed --include localhost:0,1,2,3 --master_port=20955 \
    train_decoderonly_hf.py --ds_config=${root_dir}/ds_config_hf.json \
    --train_path=${train_data_dir} \
    --valid_path=${val_data_dir} \
    --model_dir=/data/yangjunxiao/huggingface_pretrained_models/ChatGLM3/model/chatglm3-6b-base \
    --pretrained_model_path= \
    --batch_size=8 --val_batch_size=4 \
    --gradient_accumulation=1 \
    --incontext_learn=0 \
    --savedmodel_path=./hf_save/${data_name}_${method}/${model_name}_test_2 --ckpt_file='' \
    --max_epochs=${max_epoch} --warmup_steps=0 --warmup_ratio=0 \
    --learning_rate=2e-5 --fp16= \
    --seed=12 \
    --max_length=2048 --eval_step=75 --save_step=75 \
    --lr_decay=linear --patience=1 \
    --ema=0 --ema_start_epoch=0 --loss_type=GD