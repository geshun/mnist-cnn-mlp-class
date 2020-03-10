import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTClassifier(nn.Module):
    """Network achitechture for investigating MNIST dataset."""

    def __init__(self, layer_sizes, drop_prob=0.3):
        """Initializes the size of all the layers and the dropout probs.
        Defines the architecture of the network.
        Args:
            layer_sizes (list): list of integers values defining the number of
            nodes in each layer. For instance [784, 512, 128, 10] means input
            layer is of size 784, output layer of size 10 and the two hidden
            layers of sizes 512 (first hidden layer) and 128 (second hidden
            layer)
            drop_prob (float): float between 0 and 1 specifying the probability
            that a node in a layer is left out during training
        """
        self.input_layer_size = layer_sizes[0]
        self.hidden_layers_sizes = layer_sizes[1:-1]
        self.output_layer_size = layer_sizes[-1]
        self.layer_sizes = layer_sizes

        sizes = zip(layer_sizes[:-1], layer_sizes[1:])

        super().__init__()
        self.fc_layers = nn.ModuleList([])
        self.fc_layers.extend([nn.Linear(m, n) for m, n in sizes])
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """forward pass"""
        x = x.view(x.shape[0], -1)
        for layer in self.fc_layers[:-1]:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.fc_layers[-1](x)
        x = F.log_softmax(x, dim=1)
        return x

        
class MNISTConvnet(nn.Module):
    
    def __init__(self):
        """defines the network architecture"""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                              stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7*7*32, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        """forward pass"""
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

def get_accuracy_score(logits, labels):
    """Compares model output (logits) with true labels and computes
    accuracy score.
    Args:
        logits (tensor): Log probability output of model
        labels (tensor): True labels
    Returns:
        float: A numerical value as the accuracy score
    """
    probs = torch.exp(logits)
    _, top_class = torch.max(probs, 1)
    equality = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equality.float()).item()
    return accuracy


def training_loop(model, optimizer, criterion,
                  train_loader, val_loader, n_epochs=3,
                  save_model=True, save_path='checkpoint_mnist.pth'):
    """Feedforward loop.
    Args:
        model: network architecture
        optimizer: approach to minimizing the loss
        criterion: loss function to minimize
        train_loader: train data loader
        val_loader: validation data loader
        n_epochs: (int) number of epochs
        save_model: (bool) indicates to save a model or not
        save_path: (str) specifies the path of the file to save
    Returns:
    """
    if save_model:
        if type(model) == '__main__.MNISTClassifier':
            checkpoint = {'layer_sizes': model.layer_sizes,
                          'state_dict': model.state_dict()}
        else:
            checkpoint = {'state_dict': model.state_dict()}
        val_loss_min = float('Inf')

    train_losses, val_losses = [], []
    train_accuracy, val_accuracy = [], []

    for epoch in range(n_epochs):
        cum_train_loss = 0.0
        cum_train_accuracy = 0.0

        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            batch_loss = criterion(output, labels)
            batch_loss.backward()
            optimizer.step()

            cum_train_loss += batch_loss.item()
            cum_train_accuracy += get_accuracy_score(output, labels)

        else:
            avg_val_loss, avg_val_accuracy = evaluation_loop(model,
                                                             criterion,
                                                             val_loader)

        avg_train_loss = cum_train_loss/len(train_loader)
        avg_train_accuracy = cum_train_accuracy/len(train_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        train_accuracy.append(avg_train_accuracy)
        val_accuracy.append(avg_val_accuracy)

        print(f'{epoch+1}/{n_epochs} epochs\t',
              f'train_loss: {avg_train_loss:0.3f}\t',
              f'val_loss: {avg_val_loss:0.3f}\t',
              f'train_accuracy: {avg_train_accuracy:0.3f}\t',
              f'val_accuracy: {avg_val_accuracy:0.3f}')

        if save_model and avg_val_loss <= val_loss_min:
            print(f'val_loss decreased',
                  f'from {val_loss_min:0.4f} to {avg_val_loss:0.4f}\t'
                  f'saving model.....')
            checkpoint['state_dict'] = model.state_dict()
            torch.save(checkpoint, save_path)
            val_loss_min = avg_val_loss

    return (train_losses, val_losses), (train_accuracy, val_accuracy)


def evaluation_loop(model, criterion, data_loader):
    """Validates or tests model.
    Args:
        model: network architecture
        criterion: loss function to minimize
        data_loader: data loader of images and labels
    Returns: - A tuple of evalutation loss and accuracy
        avg_eval_loss (float):
        avg_eval_accuracy (float):
    """
    cum_eval_loss = 0.0
    cum_eval_accuracy = 0.0

    with torch.no_grad():
        model.eval()
        for images, labels in data_loader:
            output = model(images)
            loss = criterion(output, labels)

            cum_eval_loss += loss.item()
            cum_eval_accuracy += get_accuracy_score(output, labels)

    avg_eval_loss = cum_eval_loss/len(data_loader)
    avg_eval_accuracy = cum_eval_accuracy/len(data_loader)

    return avg_eval_loss, avg_eval_accuracy


def predict(model, X):
    """Predicts the output of a model.
    Args:
        model: trained model
        X: input of the model
    Returns:
        A tuple of top classes and corresponding probabilities.
    """
    with torch.no_grad():
        logits = model(X)
        probs = torch.exp(logits)
        top_prob, top_class = torch.max(probs, dim=1)
    return top_class, top_prob
