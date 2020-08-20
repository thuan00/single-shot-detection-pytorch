import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import torch
import torch.cuda as cuda
import cv2
import argparse
from aug import SSDInputTransform
from model import SSD300
from utils import visualize_boxes, rescale_original_coordinates
device = torch.device("cuda" if cuda.is_available() else "cpu") 

#---------------------
# Options
checkpoint_paths = {
    'ssd300': 'models/checkpoint_ssd300.pt',
    'ssdlite': 'models/ssd_mobilenetv2.pt',
    'ssdeff': 'model/ssd_efficient_net_fpn_focalloss.pt'
}

input_size = {
    'ssd300': 300,
    'ssdlite': 320,
    'ssdeff': 300
}

# Detection arguments
score_threshold=0.39
iou_threshold=0.45
top_k=100


#-----------------------------
# Arg parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False, help="model's name, can be either 'ssd300','ssdlite','ssdeff'")
ap.add_argument("-i", "--input-img", required=False, help="path to the image to be detected")
ap.add_argument("-d", "--input-dir", required=False, help="path to the directory contains images to be detected")
ap.add_argument("-o", "--output", required=True, help="path to the output folder")
args = vars(ap.parse_args())

# only accept 1 of the 2 options: input-img or input_dir
if (args['input_img'] != None) + (args['input_dir'] != None) != 1:
    raise Exception("Please provide just one of the two --input-img or --input-dir argument")


# Load model
model_name = args['model']
checkpoint = torch.load(checkpoint_paths[model_name], map_location=device)
model = checkpoint['model'].to(device)
model.eval()

# Input transform
input_transform = SSDInputTransform(size=input_size[model_name])


def detect_objects(img, score_threshold, iou_threshold, top_k):
    ''' 
    img: a single image in numpy.ndarray format, (h, w, c)
    other args: see model.py's post processing functions for more information
    return: an image with bounding boxes of detected objects drawn on it
    '''
    # input transform: 
    # -> SSDInputTransform -> transpose to channel first -> to tensor -> unsqueeze to (1,3,300,300) -> to device
    img_data, padding = input_transform(img.copy())
    img_data = torch.from_numpy(img_data.transpose((2,0,1))).unsqueeze(0).to(device) #(1,3,300,300)
    
    # inference
    predicted_offsets, predicted_scores = model(img_data)
    boxes, labels, scores = model.post_process_top_k(predicted_offsets, predicted_scores,
                                                     score_threshold, iou_threshold, top_k)
    assert (len(boxes) == 1) and (len(labels) == 1) #since batch of size 1 so the output should be a list with 1 element
    
    # output refinement
    boxes = boxes[0]
    labels = labels[0]
    if boxes.size(0) <= 0:
        return img
    h, w = img.shape[0:2]
    if h > w:
        boxes = rescale_original_coordinates(boxes, h, h)
        boxes[:,0] -= padding
        boxes[:,2] -= padding
    else:
        boxes = rescale_original_coordinates(boxes, h, w)
    boxes = boxes.tolist()
    labels = labels.tolist()
    
    return visualize_boxes(img, boxes, labels)


def detect_batch(img, score_threshold, iou_threshold, top_k):
    return


@torch.no_grad()
def main():
    if args['input_img'] is not None:
        img = cv2.imread(args['input_img'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        img = detect_objects(img, score_threshold, iou_threshold, top_k)
        end_time = time.time() - start_time
        print(f'Done in {end_time:.3f}s')
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args['output'], os.path.basename(args['input_img'])), img)
        
    elif args['input_dir'] is not None:
        img_names = [filename for filename in os.listdir(args['input_dir'])]
        img_paths = [os.path.join(args['input_dir'], filename) for filename in img_names]
        output = args['output']
        total_time = 0
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            img = detect_objects(img, score_threshold, iou_threshold, top_k)
            running_time = time.time() - start_time
            total_time += running_time

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output, os.path.basename(img_path)), img)
            
        print(f'Done in {total_time:.3f}s - FPS: {len(img_names)/total_time:.2f}')


if __name__ == '__main__':
    main()