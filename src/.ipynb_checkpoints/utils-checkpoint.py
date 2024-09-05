import cv2, torch, random, numpy as np
from collections import OrderedDict as OD
from PIL import Image
from torch.nn import functional as F
from time import time
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.metrics import Metrics 
from tqdm import tqdm
from models.unet import UNet
from models.segformer import SegFormer

def get_state_dict(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = OD()
    for k, v in checkpoint["state_dict"].items():
        name = k.replace("model.", "") # remove `model.`
        new_state_dict[name] = v
    return new_state_dict

def resize(im, im_size): return cv2.resize(im, im_size)

def tn2np(t, inv_fn=None): return (inv_fn(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8) if inv_fn is not None else (t * 255).detach().cpu().permute(2,1,0).numpy().astype(np.uint8)

def get_preds(model, test_dl, device):
    print("Start inference...")
    
    all_ims, all_preds, all_gts, acc = [], [], [], 0
    loss_fn = torch.nn.CrossEntropyLoss()
    start_time = time()
    for idx, batch in tqdm(enumerate(test_dl)):
        if idx == 10: break
        ims, gts = batch
        all_ims.extend(ims); all_gts.extend(gts);        
        preds = model(ims.to(device))
        if preds.shape[2] != gts.shape[2]: preds = torch.nn.functional.interpolate(input = preds, scale_factor = gts.shape[2] // preds.shape[2], mode = "bilinear")
        met = Metrics(preds, gts.to(device), loss_fn)
        acc += met.mIoU().item()
        all_preds.extend(preds)
        
    print(f"Inference is completed in {(time() - start_time):.3f} secs!")
    print(f"Mean Intersection over Union of the model is {acc / len(test_dl.dataset):.3f}")
    
    return all_ims, all_preds, all_gts
    
def visualize(all_ims, all_preds, all_gts, num_ims, rows, save_path, save_name):
    
    print("Start visualization...")
    plt.figure(figsize = (10, 18))
    indices = [random.randint(0, len(all_ims) - 1) for _ in range(num_ims)]
    count = 1
    threshold = -1 if "drone" in save_name else 0.5
    
    for idx, ind in enumerate(indices):
        
        im = all_ims[ind]
        gt = all_gts[ind]
        pr = all_preds[ind]
        
        plt.subplot(num_ims, 3, count)
        plt.imshow(tn2np(im.squeeze(0)))
        plt.axis("off")
        plt.title("An Input Image")
        count += 1
        
        plt.subplot(num_ims, 3, count)
        plt.imshow(tn2np(gt.unsqueeze(0)), cmap = "gray")
        plt.axis("off")
        plt.title("GT Mask")
        count += 1

        plt.subplot(num_ims, 3, count)
        plt.imshow(tn2np((pr > threshold).squeeze(0))[:, : , 1], cmap = "gray")
        plt.axis("off")
        plt.title("Generated Mask")
        count += 1
    
    plt.savefig(f"{save_path}/{save_name}_preds.png")
    print(f"The visualization can be seen in {save_path} directory.")
    
def np2tn(tfs, np): return torch.tensor(tfs(image = np)["image"]).float().permute(2, 1, 0).unsqueeze(0)

def predict(m, path, tfs, data_name, device):
    
    threshold = -1 if "drone" in data_name else 0.5
    
    im = np.array(Image.open(path))
    pred_mask = m(np2tn(tfs, im).to(device)).squeeze(0)
    res = tn2np((pred_mask > threshold).squeeze(0))[:, : , 1] if "cell" in data_name else tn2np((pred_mask > threshold).squeeze(0))[:, : , 0]
    
    return im, res

def load_pretrained_model(model_name, params, device, ckpt_path):
    model = UNet(in_chs = params["in_chs"], n_cls = params["n_cls"], out_chs = params["out_chs"], depth = params["depth"], up_method = params["up_method"]) if model_name == "unet" else \
        SegFormer(
                  in_channels=params["in_chs"],
                  widths=params["widths"],
                  depths=params["depths"],
                  all_num_heads=params["all_num_heads"],
                  patch_sizes=params["patch_sizes"],
                  overlap_sizes=params["overlap_sizes"],
                  reduction_ratios=params["reduction_ratios"],
                  mlp_expansions=params["mlp_expansions"],
                  decoder_channels=params["decoder_channels"],
                  scale_factors=params["scale_factors"],
                  num_classes=params["num_classes"],
                        )
    model = model.to(device)
    # load params
    print("\nLoading the state dictionary...")
    state_dict = get_state_dict(f"{ckpt_path}")
    model.load_state_dict(state_dict, strict=True)
    
    return model