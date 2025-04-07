import Projectplane
from OmniDepth import Pred_without_imgPath


# prepare perspective image
Panorama_path = "/home/liex/Desktop/merge/MealsRetrieval-ROOMELSA/RBG_data/Image/6_colors.png"
perspective_images = Projectplane.main(Panorama_path)
path_to_Merge = "/home/liex/Desktop/merge/MealsRetrieval-ROOMELSA/" # paste your path to Merge parent folder

# create Depth
output_path = "/home/liex/Desktop/merge/MealsRetrieval-ROOMELSA/Merged/Depth_output"  # if you need image output file detele '#' on line 83,83,99 of this file "Merged/OmniDepth/Pred_without_imgPath.py"
path_to_pretrained_model = "/home/liex/Desktop/merge/MealsRetrieval-ROOMELSA/Merged/OmniDepth/pretrained_models"

#List of Depth Matrix(1xWxH) and RBG matrix(3xWxH) 
Panorama_info = Pred_without_imgPath.main(perspective_images, output_path, path_to_pretrained_model)

print(Panorama_info[0])
