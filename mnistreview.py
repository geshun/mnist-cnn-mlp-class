import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets
from torchvision import transforms

import mnisthelper as mh
import mnistnetwork as mn

mean, std = (0.5,), (0.5,)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
trainset = datasets.MNIST('mnist-data/', train=True,
                          download=True, transform=transform)
testset = datasets.MNIST('mnist-data/', train=False,
                         download=True, transform=transform)

n_trainset = len(trainset)
val_size = 0.2
idx = torch.randperm(n_trainset)
split_at = int(val_size*n_trainset)
val_idx = idx[:split_at]
train_idx = idx[split_at:]
val_sampler = SubsetRandomSampler(val_idx)
train_sampler = SubsetRandomSampler(train_idx)

train_loader = DataLoader(trainset, batch_size=64, sampler=train_sampler)
val_loader = DataLoader(trainset, batch_size=64, sampler=val_sampler)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

images, labels = mh.get_sample(train_loader)
mh.plot_sample(images, labels)

model = mn.MNISTClassifier([784, 512, 128, 10], drop_prob=0.3)
model_cnn = mn.MNISTConvnet()
optimizer = optim.SGD(model_cnn.parameters(), lr=1e-2)
criterion = nn.NLLLoss()

(train_losses, val_losses), (train_accuracy, val_accuracy) = \
    mn.training_loop(model_cnn, optimizer, criterion, train_loader, val_loader, 5)

mh.plot_learning_curve(train_losses, val_losses)

load_model = torch.load('checkpoint_mnist.pth')
state_dict = load_model['state_dict']
layer_sizes = load_model['layer_sizes']
test_model = mn.MNISTClassifier(layer_sizes)
test_model_ = mn.MNISTConvnet()
test_model_.load_state_dict(state_dict)
mn.evaluation_loop(test_model_, criterion, test_loader)

images, labels = mh.get_sample(test_loader)
mh.plot_sample(images, mn.predict(model, images)[0])

for i in range(8, 10):
    mh.plot_model_probs(model, images, labels, i)
