import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets

import os
import copy
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from datetime import datetime

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, tb, args, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"----- Epoch {epoch} / {args.epochs} -----")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if args.save_tb:
                    tb.add_scalar("running loss", loss.item() * inputs.size(0), step)
                    tb.add_scalar("running corrects", torch.sum(preds == labels.data), step)
                    step = step + 1
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if args.save_tb:
                tb.add_scalar(f"{phase} loss", epoch_loss, epoch)
                tb.add_scalar(f"{phase} acc", epoch_acc, epoch)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"!!! New Best {best_acc:.4f} at epoch {epoch}")

        print()

    time_elapsed = time.time() - since
    hrs, mins, secs = time_elapsed // 3600,time_elapsed // 60, time_elapsed % 60
    print(f"Training complete in {hrs:.0f}h {mins:.0f}m {secs:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    model.load_state_dict(best_model_wts)
    return model

def main():
    parser = argparse.ArgumentParser(description="Food 11")
    parser.add_argument("--input", type=str, required=True, help="directory to input")
    parser.add_argument("--epochs", type=int, required=True, help="epochs for training")
    parser.add_argument("--lr", type=float, required=True, help="learning rate for training")
    parser.add_argument("--gamma", type=float, required=True, help="gamma for learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size for the dataset")
    parser.add_argument("--fine-tune", type=utils.str2bool, default=False, help="fine tune or only train last layer")
    parser.add_argument("--save-tb", type=utils.str2bool, default=True, help="save this run into tensorboard or not")
    parser.add_argument("--run-name", type=str, default=datetime.now().strftime("%Y%m%d_%H_%M_%S"), help="run name for tensor board")
    parser.add_argument("--random-seed", type=int, default=42, help="designated random seed to enabel reproducity of the training process")
    parser.add_argument("--max-data-count", type=int, default=None, help="reduce data size for quick testing")

    args = parser.parse_args()
    print("----- args -----")
    print(args)

    tb = SummaryWriter(f"runs/{args.run_name}") if args.save_tb else None

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }

    data_partitions = ["train", "val", "test"]
    image_datasets = { x: datasets.ImageFolder(os.path.join(args.input, x), data_transforms[x]) for x in data_partitions }
    dataloaders = { x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == "train" else False, num_workers=4) for x in data_partitions }
    dataset_sizes = { x: len(image_datasets[x]) for x in data_partitions }
    classes_names = image_datasets["train"].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("----- dataset sizes -----")
    print(dataset_sizes)
    print("----- classes names -----")
    print(classes_names)
    print("----- device -----")
    print(device)

    # imbalance weights
    train_distributions = np.unique(np.array(image_datasets["train"].targets), return_counts=True)[1]
    print("----- classes count -----")
    print(train_distributions)
    train_distributions = torch.Tensor(dataset_sizes["train"] / (train_distributions * 11)).to(device)
    print("----- classes weight -----")
    print(train_distributions)

    model_conv = torchvision.models.resnet50(pretrained=True)

    if not args.fine_tune:
        for param in model_conv.parameters():
            param.requires_grad = False
    
    num_features = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_features, 11)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss(weight=train_distributions)
    optimizer = optim.AdamW(model_conv.fc.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, verbose=True)

    if args.save_tb:
        images = iter(dataloaders["train"]).next()[0].to(device)
        tb.add_image("food 11", make_grid(images))
        tb.add_graph(model_conv, images)

    train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer, scheduler, tb, args, device)

    test_ground_truth = torch.Tensor([]).to(device)
    test_predictions = torch.Tensor([]).to(device)
    with torch.no_grad():
        for (images, labels) in dataloaders["test"]:
            images, labels = images.to(device), labels.to(device)
            outputs = model_conv(images)
            test_ground_truth = torch.cat((test_ground_truth, labels))
            test_predictions = torch.cat((test_predictions, torch.argmax(outputs, dim=-1)))

    confusion_df = pd.crosstab(test_ground_truth.cpu(), test_predictions.cpu(), rownames=["Actual"], colnames=["Predicted"])
    print("----- confusion matrix -----")
    print(confusion_df)
    fig, ax = plt.subplots()
    ax.matshow(confusion_df, cmap=plt.cm.Blues, alpha=0.9)
    for i, row in confusion_df.iterrows():
        for j, value in row.iteritems():
            ax.text(x=j, y=i, s=value, va='center', ha='center', size='xx-large')

    test_acc = torch.sum(test_ground_truth.cpu() == test_predictions.cpu()).double() / dataset_sizes["test"]
    print(f"test Acc: {test_acc:.4f}")
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    if args.save_tb:
        tb.add_figure("Confusion Matrix", plt.gcf())
        tb.close()

if __name__ == "__main__":
    main()