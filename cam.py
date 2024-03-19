import torch
import numpy as np
import cv2
import argparse
import os

from torchvision import models, transforms
from torch.nn import functional as F
from torch import topk

#from google.colab.patches import cv2_imshow

parser = argparse.ArgumentParser()
parser.add_argument(
    '-mp', '--model_path', default='RN18_scratch_LR0.01_Ep15',
    help='provide model file name with path'
)
parser.add_argument(
    '-ip', '--input_path', default='cam_input',
    help='provide input files folder name with path'
)
parser.add_argument(
    '-op', '--output_path', default='cam_output',
    help='provide location to store output files'
)
args = vars(parser.parse_args())

model_path = args['model_path']
input_path = args['input_path']
output_path = args['output_path']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def show_cam(CAMs, width, height, orig_image, orig_label, class_idx, all_classes, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + orig_image * 0.5
        # put class label text on the result
        cv2.putText(result, f"Predicted: {all_classes[class_idx[i]]}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(result, f"Original: {orig_label}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.imshow('CAM', result)
        cv2.imwrite(f"{output_path}/CAM_{save_name}.jpg", result)

def load_synset_classes(file_path):
    # load the synset text file for labels
    all_classes = []
    with open(file_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            labels = line.split('\n')
            current_class = labels[0]
            all_classes.append(current_class)
    return all_classes
    
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


if __name__ == '__main__':
    
    # get all the classes in a list
    all_classes = load_synset_classes('LOC_synset_mapping.txt')
    for filename in os.listdir(input_path):
        # read and visualize the image
        image = cv2.imread(f"{input_path}/{filename}")
        print(filename)
        orig_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        #get the image label
        if "NORMAL" in filename:
            orig_label = "NORMAL"
        else:
            orig_label = "PNEUMONIA"

        # load the model
        #model = models.resnet18(pretrained=True).eval()
        #model_file = 'RN18_scratch_LR0.01_Ep15'
        #model = torch.load(model_path)
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
    
        # hook the feature extractor
        # https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
        features_blobs = []
        model._modules.get('layer4').register_forward_hook(hook_feature)
        
        # get the softmax weight
        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].data.numpy())
    
        # define the transforms, resize => tensor => normalize
        data_transforms = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((224, 224)),
             transforms.ToTensor()
             #transforms.Normalize(
             #mean=[0.485, 0.456, 0.406],
            #std=[0.229, 0.224, 0.225]
        ])

        # apply the image transforms
        image_tensor = data_transforms(image)
        
        # add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        # forward pass through model
        outputs = model(image_tensor)
        
        # get the softmax probabilities
        probs = F.softmax(outputs).data.squeeze()
      
        # get the class indices of top k probabilities
        class_idx = topk(probs, 1)[1].int()

        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
      
        # file name to save the resulting CAM image with
        save_name = f"{filename.split('.')[0]}"
        print(save_name)
        #save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
     
        #show and save the results
        show_cam(CAMs, width, height, orig_image, orig_label, class_idx, all_classes, save_name)    
