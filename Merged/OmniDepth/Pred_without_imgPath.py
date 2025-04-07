import torch
import torch.nn.functional as F
from torchvision import transforms
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os.path
from pathlib import Path
import glob
import sys
import pdb
from OmniDepth.modules.unet import UNet
from OmniDepth.modules.midas.dpt_depth import DPTDepthModel
from OmniDepth.data.transforms import get_transform



def select_model(args, device, map_location, root_dir):
    print("select_model....")
    # get target task and model
    if args.task == 'depth':
        image_size = (256,512) # image size
        pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
        #model = DPTDepthModel(backbone='vitl16_384') # DPT Large
        model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location, weights_only= False)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                            # transforms.CenterCrop(image_size), # đã có size chuẩn nên không cần
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=0.5, std=0.5)])
        trans_rgb = transforms.Compose([transforms.Resize((256,512), interpolation=PIL.Image.BILINEAR),
                                                    # transforms.CenterCrop(512) # đã có size chuẩn nên không cần
                                                    ])
        return trans_totensor, model, trans_rgb
    else:
        print("task should be one of the following: normal, depth")
        sys.exit()

def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    print("standardizing.....")
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # # Standardize
    # img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(perspective_image, output_file_name, trans_totensor, model, trans_rgb, device, args):
    print("creating output......")
    with torch.no_grad():
        save_path = os.path.join(args.output_path, f'{output_file_name}_{args.task}.png')

        print('Reading input ...')
        img = perspective_image
        
        img_tensor = trans_totensor(img)[:3]

        RBG_arr = np.array(img_tensor) # image matrix
        print("RBG shape: ", RBG_arr.shape)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        # rgb_path = os.path.join(args.output_path, f'{output_file_name}_rgb.png')
        # trans_rgb(img).save(rgb_path)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        output = model(img_tensor).clamp(min=0, max=1)

        if args.task == 'depth':
            output = F.interpolate(output.unsqueeze(0), (256, 512), mode='bicubic').squeeze(0)
            output = output.clamp(0,1)
            output = 1 - output
           
            output = standardize_depth_map(output) # chuẩn hóa
            Depth_arr = np.array(output.detach().cpu())
            print("RGB-D shape: ",Depth_arr.shape)
            # plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')
            return Depth_arr, RBG_arr
        else:
            # trans_topil(output[0]).save(save_path)
            print("task error")
            sys.exit()
            
        print(f'Writing output {save_path} ...')

def get_infor(output_path, root_dir):
    print("get infor")
    parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')
    parser.add_argument('--task', dest='task', help="normal or depth")
    parser.set_defaults(task='NONE')
    parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
    parser.set_defaults(store_name='NONE')

    sys.argv = [
        'test.py',
        '--task', 'depth',
        '--output_path', output_path # output dir
    ]
    root_dir = root_dir + "/" # change to "\" if you on windows
    args = parser.parse_args()
    return args, root_dir


def main(perspective_images, output_path, pretraied_model_path):

    args, root_dir = get_infor(output_path, pretraied_model_path) # input to your dir

    # create output folder, create map, set devide(cuda or cpu)
    os.system(f"mkdir -p {args.output_path}")
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #load model
    trans_totensor, model, trans_rgb = select_model(args, device, map_location, root_dir)
    # trans_topil = transforms.ToPILImage()

    # load images, create output
    id = 1
    Panoramic_info = []
    try:
        for image in perspective_images:
            
            Depth_arr, RBG_arr = save_outputs(image, "Perspective_Depth" + str(id), trans_totensor, model, trans_rgb, device, args)
            Perspective_info = {
                f"Perspective_RGB_array {str(id)}" : RBG_arr,
                f"Perspective_Depth_array {str(id)}" : Depth_arr
            }
            Panoramic_info.append(Perspective_info)
            id += 1
        return Panoramic_info
    except:
        print("Perspective_images error")
