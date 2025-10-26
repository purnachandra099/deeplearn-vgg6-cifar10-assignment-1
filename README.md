# CS6886W - VGG6 CIFAR-10 Experiments

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run baseline: `python vgg6_baseline.py`
3. Run W&B sweep: `wandb sweep sweep_config.yaml`

## Environment
- Python 3.10
- PyTorch 2.2+
- torchvision 0.17+
- wandb

## Reproducibility
- Seed = 42
- Best configuration: GELU + Nadam + batch 128 + lr 0.001

## Results
Validation accuracy ≈ 88.5 %
