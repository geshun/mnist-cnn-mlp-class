import torch
import matplotlib.pyplot as plt
import numpy as np

from mnistnetwork import predict

def get_sample(data_loader):
    """Returns sample of images and labels as tuple."""
    batch_sample = next(iter(data_loader))
    return batch_sample


def plot_sample(images, labels=None, n_samples=20, nrow=2, ncol=10):
    """Plots a specified sample of mnist data.
    Args:
        images: (tensor) images to plot
        labels: (tensor) corresponding label of images
        n_samples: (int) number of samples to plot. Can't exceed batch size
        nrow: (int) height of plot figure
        ncol: (int) width of plot figure
    """
    fig = plt.figure(figsize=(ncol, nrow+0.5))
    for i in range(n_samples):
        image = images[i].squeeze().numpy()
        ax = fig.add_subplot(nrow, ncol, i+1, xticks=[], yticks=[])
        ax.imshow(image, cmap='gray')
        if labels is not None:
            label = str(labels[i].item())
            ax.set_title(label)
    plt.show()


def plot_learning_curve(train_losses, val_losses):
    """Plots learning curve of data gathered during training.
    Args:
        train_losses: (list) float numbers of train loss at each epoch
        val_losses: (list) float numbers of validation loss at each epoch
    """
    n_epochs = len(train_losses)
    epochs = list(range(1, n_epochs+1))

    plt.plot(epochs, train_losses, label='training')
    plt.plot(epochs, val_losses, label='validation')
    plt.grid(True)
    plt.legend()
    plt.title('Learning Curve')
    plt.xlabel('num of epochs')
    plt.ylabel('crossentropy loss')
    plt.show()
    
    
def plot_model_probs(model, images, labels, i=0):
    """Plots the true image and the probabilities of the predicted image.
    Args:
        model: trained model
        images: (tensor)
        labels: (tensor)
        i: (int) index of the image
    """
    with torch.no_grad():
        logits = model(images[i])
        probs = torch.exp(logits)
    image = images[i].squeeze().numpy()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title(f'True label: ' + str(labels[i].item()))
    ax[1].bar(np.arange(10), probs[0])
    ax[1].set_title(f'model predicts: '+str(predict(model, images[i])[0].item()))
    plt.xticks(np.arange(10))
    plt.show()
