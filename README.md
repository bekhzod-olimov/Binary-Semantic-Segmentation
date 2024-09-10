# Semantic Segmentation Projects

This repository contains a deep learning (DL)-based artificial intelligence (AI) semantic segmentation model, namely, [SegFormer](https://github.com/NVlabs/SegFormer) and [UNet](https://github.com/milesial/Pytorch-UNet) training using various datasets, such as, [Cells](https://www.kaggle.com/datasets/killa92/medical-cells-image-segmentation), [Flood](https://www.kaggle.com/datasets/killa92/flood-image-segmentation), [Drone](https://www.kaggle.com/datasets/killa92/drone-images-semantic-segmentation) Segmentation datasets.

# These are the steps to use this repository:

1. Clone the repository:

`git clone https://github.com/bekhzod-olimov/Binary-Semantic-Segmentation.git`

`cd Binary-Semantic-Segmentation`

2. Create conda environment and activate it using the following script:
   
`conda create -n ENV_NAME python=3.10`

`conda activate ENV_NAME`

(if necessary) add the created virtual environment to the jupyter notebook (helps with debugging)

`python -m ipykernel install --user --name ENV_NAME --display-name ENV_NAME`

3. Train the pre-defined AI models using the following script:

Train process arguments can be changed based on the following information:

![image](https://github.com/user-attachments/assets/286abcef-d23e-4ff2-9479-b45178fea479)

a) UNet

```python
python train.py --model_name unet --batch_size 4 devices 2 --epochs 30
```

a) SegFormer

```python
python train.py --model_name segformer --batch_size 8 devices 3 --epochs 20
```

4. Conduct inference using the pre-trained models:
a) UNet

```python
python inference.py --model_name unet --save_path inference_results
```

a) SegFormer

```python
python inference.py --model_name segformer --save_path results
```
