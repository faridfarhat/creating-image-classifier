
import torch
import argparse
from torchvision import transforms
from PIL import Image

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type = str, help = 'path to the folder of images')
    parser.add_argument('checkpoint', type = str, help='path to saved model')
    parser.add_argument('--top_k',type = int, default= 3, help = 'define number of classess for prediction')
    parser.add_argument('--category_names',type = str, default= '/home/workspace/ImageClassifier/', help = 'categories')
    parser.add_argument('--gpu', action = 'store_true', help= 'gpu')
    return parser.parse_args()

in_arg = get_input_args()
image_path = in_arg.image_path
checkpoint_dir = in_arg.checkpoint
no_top_k = in_arg.top_k
category_names_dict = in_arg.category_names

gpu = in_arg.gpu

if gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import json
with open(category_names_dict, 'r') as f:
    cat_to_name = json.load(f)

    

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model
    
model=load_checkpoint(checkpoint_dir)


def process_image(image):
    transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])                            

    pic = Image.open(image)
    pic = transform(pic).float()
    pic = pic.unsqueeze_(0)
    pic = pic.to(device)
    return pic


def predict(image_path, model, topk=no_top_k):

    image = process_image(image_path)
    
    with torch.no_grad():
        model.eval()
        logs = model(image)
        ps = torch.exp(logs)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p, top_class = top_p.flatten().tolist(), top_class.flatten().tolist()
        
        classes=[]
        for i in top_class:
            tup = tuple(model.class_to_idx.items())
            classes.append(tup[i][0])

        return top_p, classes



x= predict(image_path, model)[0]

y= predict(image_path, model)[1]

flower=[]

for i in y:
    if i in cat_to_name:
        flower.append(cat_to_name[i])
    if i not in cat_to_name:
        flower.append('unkown class')


print(flower)
print(x)


            
