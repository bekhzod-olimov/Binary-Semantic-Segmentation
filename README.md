# Semantic Segmentation Projects

This repository contains a deep learning (DL)-based artificial intelligence (AI) semantic segmentation model, namely, [SegFormer](https://github.com/NVlabs/SegFormer) and [UNet](https://github.com/milesial/Pytorch-UNet) training using various datasets, such as, [Cells](https://www.kaggle.com/datasets/killa92/medical-cells-image-segmentation), [Flood](https://www.kaggle.com/datasets/killa92/flood-image-segmentation), [Drone](https://www.kaggle.com/datasets/killa92/drone-images-semantic-segmentation) Segmentation datasets.

# Inference results using the pretrained models are as follows:

- UNet Model pretrained on Cells Dataset

![cells_unet_preds](https://github.com/user-attachments/assets/cf3314e0-60f1-4da7-992f-a604f95844c2)

- UNet Model pretrained on Flood Dataset

![flood_unet_preds](https://github.com/user-attachments/assets/ce888acb-b00b-4ca8-a973-2df51d2a275d)

- UNet Model pretrained on Drone Dataset

![drone_unet_preds](https://github.com/user-attachments/assets/901af34e-9d54-444d-8491-a5f457176284)

- SegFormer Model pretrained on Cells Dataset

![cells_segformer_preds](https://github.com/user-attachments/assets/d65f70cf-c9a9-40f2-a120-e8b3c37045ac)

- SegFormer Model pretrained on Flood Dataset

![flood_segformer_preds](https://github.com/user-attachments/assets/f1e759ba-1b26-4010-b135-f6885fdfbdd9)

- SegFormer Model pretrained on Drone Dataset

![drone_segformer_preds](https://github.com/user-attachments/assets/ceaa9149-2a6f-44a6-83d3-c2abc34a7ee0)

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

![image](https://github.com/user-attachments/assets/b1444449-8c89-4f8a-a632-268f2676ff6e)

![image](https://github.com/user-attachments/assets/c4b5f5ec-aeb0-41ef-94d0-d8faec0b8c56)

![image](https://github.com/user-attachments/assets/647d7181-5ed4-4e79-b6c4-1b51a7f327a5)

b) SegFormer

```python
python train.py --model_name segformer --batch_size 8 devices 3 --epochs 20
```

![image](https://github.com/user-attachments/assets/ff4755e9-c367-4139-8d82-bce65a3b60c0)

![image](https://github.com/user-attachments/assets/bb89fd26-2e6b-400a-a2b7-c12cf510b9b8)

![image](https://github.com/user-attachments/assets/e35ca015-919b-419d-85db-58726cc2ab06)

4. Conduct inference using the pre-trained models:

Inference process arguments can be changed based on the following information:

![image](https://github.com/user-attachments/assets/310239f3-b4bb-4533-9928-0fdd6a91c872)

All pretrained models using the aforementioned datasets are publicly available and downloaded during the inference process.

a) UNet

```python
python inference.py --model_name unet --save_path inference_results
```

b) SegFormer

```python
python inference.py --model_name segformer --save_path results
```

5. Demo using the pretrained AI models:

Demo script arguments can be changed based on the following information:

![image](https://github.com/user-attachments/assets/22a3c7b6-35ae-4174-b04c-183ab712a6bc)

All pretrained models using the aforementioned datasets are publicly available and downloaded during the inference process.

```python
streamlit run demo.py
```
