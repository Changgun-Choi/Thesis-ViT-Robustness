# Thesis_Vision
## Thesis for Master's degree at Mannheim University (ChangGun Choi)

## Data
In order to run the code from this repository you run ImageNet.py to download validation set. 
It consists of several data folders that are used for different parts of this project.

## Structure of VisionTransformer
cd "/home/cchoi/data/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer"
```
VisionTransformer/
├── vit_foolbox_robust_original.py
├── vit_foolbox_frequency.py
├── vit_foolbox_robust_Calibration
├── vit_explain_gradcam
└── vit_explain_ViT
```
## Requirements
cd "/home/cchoi/data/Thesis_Vision"
To install them, run: pip install -r Thesis_Vision/requirements.txt

## Run
cd "/home/cchoi/data/Thesis_Vision/VisionTransformer/VisionTransformer/VisionTransformer" 
### Adversarial attack(White-box)
python vit_foolbox_robust_original.py --model_name vit --attack_name PGD --batch_size 16 --data_divide 62  --data_path full_server
### Frequency Filter applied to Adversarial attack
1. High-frequency

python vit_foolbox_frequency.py --model_name vit_hybrid --attack_name PGD --batch_size 16 --data_divide 10 --filter y --data_path full_server --filter_f high

2. Low-frequency

python vit_foolbox_frequency.py --model_name vit_hybrid --attack_name PGD --batch_size 16 --data_divide 10 --filter y --data_path full_server --filter_f low
### Calibration Error Analysis
python vit_foolbox_robust_Calibration.py --model_name efficient_b4 --attack_name PGD --batch_size 16 --data_divide 10 --data_path full_server 
### Vit Attention Rollout (download image and create the path)
python vit_explain.py --image_path "/image.png" --head_fusion "min" --discard_ratio 0.8 
### Grad Cam (set category index for the image)
python vit_explain_gradcam.py --model_name VGG --attack_name LinfPGD --visual Grad_Cam --use_cuda --category_index 726
