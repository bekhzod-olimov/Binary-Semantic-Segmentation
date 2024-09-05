# Import libraries
import os, torch, sys, pickle, argparse, numpy as np, streamlit as st
from glob import glob
from torchvision.datasets import ImageFolder
from src.utils import load_pretrained_model, predict, np2tn, resize
from models.params import get_params
from src.transformations import get_transformations
from PIL import Image
sys.path.append("./")
st.set_page_config(layout='wide')
sys.path.append(os.getcwd())

def run(args):
    
    """
    
    This function gets parsed arguments and runs the script.
    
    Parameter:
    
        args   - parsed arguments, argparser object;
        
    """
    
    checkpoint_path = f"{args.save_model_path}/{args.model_name}_{args.dataset_name}_best.ckpt"

    # Initialize transformations to be applied
    tfs = get_transformations(args.inp_im_size)[1]
    # Set a default path to the image
    default_path = glob(f"{args.root}/{args.dataset_name}/*.jpg")[1]
    fname = os.path.splitext(os.path.basename(default_path))[0]
    gt = Image.open(default_path.replace(fname, f"{fname}_gt").replace(".jpg", ".png"))
    
    # Load segmentation model
    params = get_params(args.model_name)
    ckpt_path = f"{args.save_model_path}/{args.model_name}_{args.dataset_name}_best.ckpt"
    m = load_pretrained_model(model_name = args.model_name, params = params, device = args.device, ckpt_path = ckpt_path)
    print(f"The {args.model_name} state dictionary is successfully loaded!\n")

    st.title(f"AI 세그멘테이션 모델 데모")
    file = st.file_uploader('이미지를 업로드해주세요')
    # Get image and predicted class
    inp = file if file else default_path
    im, out = predict(m = m, path = inp, tfs = tfs, data_name = args.dataset_name, device = args.device)
    
    col1, col2, col3 = st.columns(3)
    
    with col1: st.header("입력된 이미지:"); st.image(resize(im, gt.size))
    with col2: st.header("입력된 의료 마스크:"); st.image(gt)
    with col3: st.header("AI 생성한 마스크:"); st.image(resize(out, gt.size))
    
if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = "Semantic Segmentation Demo")
    
    # Add arguments
    parser.add_argument("-r", "--root", type = str, default = "sample_ims", help = "Root folder for test images")
    parser.add_argument("-dn", "--dataset_name", type = str, default = 'drone', help = "Dataset name for training")
    parser.add_argument("-is", "--inp_im_size", type = int, default = 320, help = "Input image size")
    parser.add_argument("-mn", "--model_name", type = str, default = 'unet', help = "Model name for backbone")
    parser.add_argument("-d", "--device", type = str, default = 'cpu', help = "GPU device name")
    parser.add_argument("-sm", "--save_model_path", type = str, default = 'saved_models', help = "Path to the directory to save a trained model")
    parser.add_argument("-dp", "--data_path", type = str, default = "saved_dls", help = "Dataset name")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args) 