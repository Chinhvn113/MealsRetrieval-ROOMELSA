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
from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform



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
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(img_path, output_file_name, trans_totensor, model, trans_rgb, device, args):
    print("creating output......")
    with torch.no_grad():
        save_path = os.path.join(args.output_path, f'{output_file_name}_{args.task}.png')

        print(f'Reading input {img_path} ...')
        img = Image.open(img_path)

        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

        rgb_path = os.path.join(args.output_path, f'{output_file_name}_rgb.png')
        trans_rgb(img).save(rgb_path)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        output = model(img_tensor).clamp(min=0, max=1)

        if args.task == 'depth':
            output = F.interpolate(output.unsqueeze(0), (256, 512), mode='bicubic').squeeze(0)
            output = output.clamp(0,1)
            output = 1 - output
           
            output = standardize_depth_map(output) # chuẩn hóa

            arr = np.array(output.detach().cpu().squeeze())
            print ( arr.shape)
            print(arr)
            plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')
            return arr
        else:
            # trans_topil(output[0]).save(save_path)
            print("task error")
            sys.exit()
            
        print(f'Writing output {save_path} ...')

def get_infor(img_path, output_path, root_dir):
    print("get infor")
    parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')
    parser.add_argument('--task', dest='task', help="normal or depth")
    parser.set_defaults(task='NONE')
    parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
    parser.set_defaults(im_name='NONE')
    parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
    parser.set_defaults(store_name='NONE')

    sys.argv = [
        'test.py',
        '--task', 'depth',
        '--img_path', img_path,  # image dir
        '--output_path', output_path # output dir
    ]
    root_dir = root_dir + "/" # change to "\" if you on windows
    args = parser.parse_args()
    return args, root_dir


def main(img_path, output_path, pretraied_model_path):

    args, root_dir = get_infor(img_path, output_path, pretraied_model_path) # input to your dir

    # create output folder, create map, set devide(cuda or cpu)
    os.system(f"mkdir -p {args.output_path}")
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #load model
    trans_totensor, model, trans_rgb = select_model(args, device, map_location, root_dir)
    # trans_topil = transforms.ToPILImage()

    # load images, create output
    img_path = Path(args.img_path)
    if img_path.is_file():
        Depth_arr = save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0], trans_totensor, model, trans_rgb, device, args)
    elif img_path.is_dir():
        # will change if need Depth Matrix
        for f in glob.glob(args.img_path+'/*'):
            Depth_arr = save_outputs(f, os.path.splitext(os.path.basename(f))[0], trans_totensor, model, trans_rgb, device, args)
    else:
        print("invalid file path!")
        sys.exit()

