# YOLOX - Training on a Custom COCO Dataset

This project is configured to train the **YOLOX-S** model on a custom dataset in COCO format.

## Installation

First, clone the repository and install the required dependencies.

```shell
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -v -e .
```

## Custom Dataset Training

Follow these steps to train the YOLOX-S model on your custom dataset.

### 1. Prepare Your Dataset

Ensure your custom dataset is in the COCO format. Then, create a symbolic link from your dataset directory to `datasets/COCO`.

```shell
# Navigate to the YOLOX project root
cd /path/to/your/YOLOX

# Create a 'datasets' directory if it doesn't exist
mkdir -p datasets

# Link your COCO dataset
ln -s /path/to/your/custom_coco_dataset ./datasets/COCO
```

Your dataset directory should have the following structure:
```
COCO/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
│   ├── image1.jpg
│   └── ...
└── val2017/
    ├── image2.jpg
    └── ...
```

### 2. Start Training

Use the following command to start training the `yolox-s` model. The command uses the experiment file `exps/default/yolox_s.py`.

```shell
python -m yolox.tools.train \
    -f exps/default/yolox_s.py \
    -d 1 \
    -b 8 \
    --fp16 \
    -o
```

**Command Arguments:**
*   `-f`: Path to the experiment file for the model you want to train.
*   `-d`: Number of GPU devices to use for training.
*   `-b`: Total batch size across all GPUs (e.g., `num_gpu * 8`).
*   `--fp16`: Enables mixed-precision training to speed up the process and reduce memory usage.
*   `-o`: (Optional) Overwrites the experiment config.

### 3. Evaluation

To evaluate your trained model on the validation set, use the `eval.py` script. Make sure to replace `path/to/your/trained_model.pth` with the actual path to your saved model checkpoint.

```shell
python -m yolox.tools.eval \
    -f exps/default/yolox_s.py \
    -c /path/to/your/trained_model.pth \
    -b 8 \
    -d 1 \
    --conf 0.001
```

**Command Arguments:**
*   `-c`: Path to the trained model weight file (`.pth`).
*   `--conf`: Confidence threshold for evaluation.

## Cite YOLOX
If you use YOLOX in your research, please cite the original paper:
```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```