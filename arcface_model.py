import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2

from loss import ArcFace

class ArcFaceModel(nn.Module):
    """
    EfficientNet V2 - based model that has been adapted to be used with ArcFace loss.
    """

    def __init__(self, num_classes):
        """
        Creates an ArcFaceModel. It combines an extractor (EfficientNet V2 S without the classifier but with a dropout and fully-connected layer instead that produces an embedding of size 512)
        and an additional fully-connected layer that maps the embedding to a given number of classes.

        :param num_classes: Number of classes
        """

        super(ArcFaceModel, self).__init__()
        self.efficientnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        self.extractor = nn.Sequential(
            self.efficientnet,
            nn.Dropout(0.5),
            nn.Linear(in_features, 512)
        )

        self.fc = nn.Linear(512, num_classes)
        self.returnEmbedding = False

    def forward(self, x):
        x = F.normalize(self.extractor(x))

        if self.returnEmbedding:
            return x
        else:
            logits = F.linear(x, F.normalize(self.fc.weight))
            return logits
    
# Training parameters
IMG_SIZE = 384
BATCH_SIZE = 16

LEARNING_RATE = 0.00001
LEARNING_RATE_DECAY = 0.15

MAX_EPOCHS = 50
EARLY_STOPPING_THRESHOLD = 0.15 # stops training when training loss is lower than this constant
    
def train_model(device, num_classes, trainloader):
    """
    Trains a model.

    :param device: (CUDA) device to use
    :param num_classes: number of classes
    :param trainloader: PyTorch dataloader that contains the training data
    """

    print("initialize model")
    model = ArcFaceModel(num_classes=num_classes)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=(1-LEARNING_RATE_DECAY))

    loss_function = nn.CrossEntropyLoss()
    arcface_loss = ArcFace()

    model.to(device)

    running_loss = 0
    train_losses = []

    start = time.perf_counter()

    print(f"start training for maximum {MAX_EPOCHS} epochs")

    for epoch in range(MAX_EPOCHS):
        model.train()

        for images, labels, _ in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images)
            arc_logits = arcface_loss(logits.clone(), labels)
            loss = loss_function(arc_logits, labels)

            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        train_losses.append(running_loss/len(trainloader))
    
        now_mins = int((time.perf_counter()-start)/60)

        print(f"{now_mins} mins -> Epoch {epoch+1}/{MAX_EPOCHS}.. Train loss: {running_loss/len(trainloader):.4f}, Learning rate: {scheduler.get_last_lr()}")
    
        if len(train_losses) >= 3:
            if train_losses[-1] < EARLY_STOPPING_THRESHOLD and train_losses[-2] < EARLY_STOPPING_THRESHOLD and train_losses[-3] < EARLY_STOPPING_THRESHOLD:
                print(f"early stopping due to loss < {EARLY_STOPPING_THRESHOLD}")
                break
    
        running_loss = 0

        scheduler.step()

    return model

def compose_input_transform(mean, std, image_target_size):
    """
    Composes a torchvision input transform based on the mean and standard deviation of the pixel values in the data set, and the image target size.

    :param mean: mean of pixel values for normalization.
    :param std: std of pixel values for normalization.
    :param image_target_size: target size of the images that are fed through the network. Will be resized to a square.
    """

    return v2.Compose([
        v2.Resize(image_target_size),
        v2.PILToTensor(),
        v2.ConvertImageDtype(torch.float),
        v2.Normalize(mean=mean, std=std),
    ])