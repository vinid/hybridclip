# Hybrid CLIP

Slightly edited version of the hybrid CLIP model implemented here:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/contrastive-image-text/run_clip.py


```bash
pip install git+https://github.com/huggingface/transformers
git clone https://github.com/vinid/hybridclip
pip install -r requirements
```

```bash

python  hybridclip/main_script.py \
    --output_dir ./clip-roberta-finetuned \
    --model_name_or_path ./medical_clip \
    --train_file location_of_train.csv \
    --validation_file location_of_test.csv \
    --image_column image_path \
    --caption_column caption \
    --remove_unused_columns=False \
    --evaluation_strategy="steps" \
    --do_train  --do_eval \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --num_train_epochs 10 \
    --learning_rate="5e-5"  \
    --warmup_steps="0"  \
    --weight_decay 0.1 \
    --overwrite_output_dir \
    --report_to="wandb" \
    --wandb = "key" \
    --vision_encoder="openai/clip-vit-base-patch32" \
    --text_encoder="emilyalsentzer/Bio_ClinicalBERT"

```