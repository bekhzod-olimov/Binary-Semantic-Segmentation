import torch, yaml, sys, os, pickle, timm, argparse
from src.utils import get_state_dict, get_preds, visualize, load_pretrained_model
from models.params import get_params
sys.path.append("./src")

def run(args):
    
    """
    
    This function runs the infernce script based on the arguments.
    
    Parameter:
    
        args - parsed arguments.
        
    Output:
    
        train process.
    
    """
    
    assert args.dataset_name in ["flood", "cells", "drone", "isic"], "Please choose the proper dataset name"
    
    # Get train arguments 
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n\n{argstr}")
    
    os.makedirs(args.save_path, exist_ok=True)
    
    params = get_params(args.model_name)
    test_dl = torch.load(f"{args.dls_dir}/{args.dataset_name}_test_dl")
    print(f"Test dataloader is successfully loaded!")
    print(f"There are {len(test_dl)} batches in the test dataloader!")
    
    ckpt_path = f"{args.save_model_path}/{args.model_name}_{args.dataset_name}_best.ckpt"
    model = load_pretrained_model(model_name = args.model_name, params = params, device = args.device, ckpt_path = ckpt_path)
    print(f"The {args.model_name} state dictionary is successfully loaded!\n")
    all_ims, all_preds, all_gts = get_preds(model, test_dl, args.device)
    
    visualize(all_ims, all_preds, all_gts, num_ims = 10, rows = 2, save_path = args.save_path, save_name = f"{args.dataset_name}_{args.model_name}")
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = 'Semantic Segmentation Training Arguments')
    
    # Add arguments to the parser
    parser.add_argument("-dn", "--dataset_name", type = str, default = 'drone', help = "Dataset name for training")
    parser.add_argument("-mn", "--model_name", type = str, default = 'unet', help = "Model name for backbone")
    parser.add_argument("-d", "--device", type = str, default = 'cuda:1', help = "GPU device name")
    # parser.add_argument("-mn", "--model_name", type = str, default = 'vit_base_patch16_224', help = "Model name for backbone")
    # parser.add_argument("-mn", "--model_name", type = str, default = 'vgg16_bn', help = "Model name for backbone")
    parser.add_argument("-sm", "--save_model_path", type = str, default = 'saved_models', help = "Path to the directory to save a trained model")
    parser.add_argument("-sp", "--save_path", type = str, default = "results", help = "Path to dir to save inference results")
    parser.add_argument("-dl", "--dls_dir", type = str, default = "saved_dls", help = "Path to dir to save dataloaders")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)