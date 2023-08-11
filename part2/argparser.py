import argparse
 
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type = str, help = 'path to the folder of images')
    parser.add_argument('--save_dir', type = str, default='/home/workspace/saved_models/', help='path to saved model')
    parser.add_argument("--arch", type=str, default="vgg13", choices=["vgg13", "resnet18", "densenet121"], help="Name of the model to load (vgg13, resnet18, densenet121)")
    parser.add_argument('--learning_rate',type = float, default= 0.01, help = 'learning rate')
    parser.add_argument('--hidden_units',type = int, default= 512, help = 'hidden units')
    parser.add_argument('--epochs',type = int, default= 20, help = 'epochs')
    parser.add_argument('--gpu', action = 'store_true', help= 'gpu')
    return parser.parse_args()
