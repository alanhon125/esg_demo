# PubTables-1M

![](https://user-images.githubusercontent.com/24642166/150664500-c8a8359b-12b0-4ea7-be8b-12f6cc773fd4.png)

- Quick Start: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CePiqlZfJa_tLTCzbatOKahWlyEAMgQa?usp=sharing)

- Pre-trained weights: [Link](https://drive.google.com/drive/folders/1Ko4Trk48u99AAPNU41RcUKAoMP0BoDmU?usp=sharing)


## Update: 
- Mar 14 2022: 
  - Release pre-trained weights (20 epochs on full training set)
- Mar 5 2022: 
  - Release pre-trained weights (12 epochs on full training set)
- Mar 1 2022: 
  - Release pre-trained weights (7 epochs on full training set)
  - Update core.py and google colab for new code
- Jan 23 2022:
  - Release pre-trained weights (20 epochs on a small set of data)
- Jan 17 2022: 
  - Release pre-trained weights (11 epochs on a small set of data).
  - Add docker training
  - Add streamlit 

## Evaluation of pre-trained epoches

### Epoch 20 - training on full training set

```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.912
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.970
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.947
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.709
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.910
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.916
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.445
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.861
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.941
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.763
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.938
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.947
pubmed: AP50: 0.970, AP75: 0.947, AP: 0.912, AR: 0.941
Total training time:  18 days, 15:03:14.605917
```


## For Docker users

```bash
docker pull phamquiluan/table-transformer:latest
# or
docker build -t phamquiluan/table-transformer -f Dockerfile .

# train TSR
docker run -it --shm-size 8G --gpus all -v /data/pubtables1m/PubTables1M-Structure-PASCAL-VOC:/code/data phamquiluan/table-transformer /bin/bash
cd src; python3 main.py --data_root_dir /code/data --data_type structure
```


## Code Installation
Create a virtual environment and activate it as follows
```
python3.7 -m venv env; source env/bin/activate
pip install -U pip

pip install -r requirements.txt
```

## For streamlit users

```bash 
streamlit run app.py
```

[Original README](https://github.com/microsoft/table-transformer)
