from argparser import get_input_args
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

in_arg = get_input_args()
data_dir = in_arg.data_dir
checkpoint_dir = in_arg.save_dir
input_model = in_arg.arch
learning_rate = in_arg.learning_rate
hidden_units = in_arg.hidden_units
epochs = in_arg.epochs
gpu = in_arg.gpu

device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.229, 0.224, 0.225],
                                                            [0.485, 0.456, 0.406])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.229, 0.224, 0.225],
                                                            [0.485, 0.456, 0.406])])

train_dataset = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
valid_dataset = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)


trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64)

print(input_model)


vgg_classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.2)),
    ('fc2', nn.Linear(hidden_units, 102)),
    ('logsoftmax', nn.LogSoftmax(dim=1))
]))

dense_classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.2)),
    ('fc2', nn.Linear(hidden_units, 102)),
    ('logsoftmax', nn.LogSoftmax(dim=1))
]))


if input_model == 'vgg13':
    model = models.vgg13(pretrained=True)
    for para in model.parameters():
        para.requires_grad = False
    model.classifier = vgg_classifier
elif input_model == 'densenet121':
    model = models.densenet121(pretrained=True)
    for para in model.parameters():
        para.requires_grad = False
    model.classifier = dense_classifier
else:
    raise ValueError("Unsupported model name. Please choose from 'vgg13', or 'densenet121'.")


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

model.to(device)

train_losses = []
valid_losses = []

for epoch in range(epochs):

    train_loss = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        log = model(images)
        loss = criterion(log, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    else:
        valid_loss = 0
        accuracy = 0
        model.eval()

        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)

                log = model(images)
                loss = criterion(log, labels)
                valid_loss += loss.item()

                ps = torch.exp(log)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        train_losses.append(train_loss/len(trainloader))
        valid_losses.append(valid_loss/len(validloader))

        model.train()

        print(f'epoch ({epoch+1}/{epochs}), training loss ({train_loss/len(trainloader):.3f}), validation loss ({valid_loss/len(validloader):.3f}), accuracy ({accuracy/len(validloader):.3f})')

 
model.class_to_idx = full_dataset.class_to_idx
check_points = {'model': model,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}

torch.save(check_points, checkpoint_dir + 'checkpoint.pth')

