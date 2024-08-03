# General Table Extraction

A general tool for table extraction from pdf files.

# Table Detection

## Environment Preparation

Use conda environment detectron2 on the AI server.

Or install the following libraries (refer to https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

1. Python 3.7
2. PyTorch 1.8 and torchvision that matches the PyTorch installation.
3. OpenCV

## Table Detection Model

Get the pretrained model from http://10.6.55.100:5000/sharing/iSJ73b7bL or https://github.com/doc-analysis/TableBank/blob/master/MODEL_ZOO.md.

Put the model file model_final_X101.pth and the config file All_X101.yml to the detectron2/demo folder.

## Run Table Detection Model

Run the following command to detect tables in an image.

python demo.py --config-file All_X101.yml \
  --input test3.png \
  --output test3_r.png \
  --opts MODEL.WEIGHTS model_final_X101.pth

# Table Structure Generation

TBD (We can use Camelot for now)
