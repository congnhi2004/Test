
from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine, ColorJitter
import torch.nn as nn
from torchsummary import summary
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, \
    FasterRCNN_ResNet50_FPN_V2_Weights
import numpy as np
import cv2
import argparse
import os
import shutil
from tqdm.autonotebook import tqdm
import warnings
warnings.filterwarnings("ignore")


def collate_fn(batch):
    images = []
    targets = []

    for i, t in batch:
        images.append(i)
        targets.append(t)

    return images, targets


def get_args():
    parser = argparse.ArgumentParser(description="Train an object detector")
    parser.add_argument("--data-path", "-d", type=str, default="data/pascal_voc")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--image-size", "-i", type=int, default=416)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--log_path", "-l", type=str, default="pascal_voc")
    parser.add_argument("--save_path", "-s", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()
    return args


class PASCALVOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform, size=1000):
        super().__init__(root=root, year=year, image_set=image_set, download=download, transform=transform)
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']
        self.size = size

    def __getitem__(self, index):
        image, data = super().__getitem__(index)
        boxes = []
        labels = []
        ori_width = int(data["annotation"]["size"]["width"])
        ori_height = int(data["annotation"]["size"]["height"])
        for obj in data["annotation"]["object"]:
            bbox = obj["bndbox"]
            xmin = int(bbox["xmin"]) / ori_width * self.size
            ymin = int(bbox["ymin"]) / ori_height * self.size
            xmax = int(bbox["xmax"]) / ori_width * self.size
            ymax = int(bbox["ymax"]) / ori_height * self.size
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes.index(obj["name"]))
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        targets = {"boxes": boxes, "labels": labels}
        return image, targets


if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_transform = Compose([
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    test_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    train_set = PASCALVOCDataset(root=args.data_path, year="2012", image_set="train", download=False,
                                 transform=train_transform, size=args.image_size)
    test_set = PASCALVOCDataset(root=args.data_path, year="2012", image_set="val", download=False,
                                transform=test_transform, size=args.image_size)

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn
    )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                                                 trainable_backbone_layers=0)
    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=20 + 1, bias=True)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=4 * (20 + 1), bias=True)
    model.to(device)

    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.mkdir(args.log_path)
    writer = SummaryWriter(args.log_path)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = []
        progress_bar = tqdm(train_dataloader, colour="cyan")
        for i, (images, targets) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            # final_targets = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)} for target in targets]
            final_targets = []
            for t in targets:
                target = {}
                target["boxes"] = t["boxes"].to(device)
                target["labels"] = t["labels"].to(device)
                final_targets.append(target)
            output = model(images, final_targets)
            loss_value = 0
            for l in output.values():
                loss_value = loss_value + l

            optimizer.zero_grad()
            loss_value.backward()
            train_loss.append(loss_value.item())
            optimizer.step()
            progress_bar.set_description(
                "Epoch {}. Iteration {}/{} Loss {:0.5f}".format(epoch + 1, i + 1, len(train_dataloader),
                                                                np.mean(train_loss)))
            writer.add_scalar("Train/Loss", np.mean(train_loss), i + epoch * len(train_dataloader))

        if os.path.isdir(args.save_path):
            shutil.rmtree(args.save_path)
        os.mkdir(args.save_path)
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.save_path, "faster_rcnn_{}.pt".format(epoch)))

        # model.eval()
        # val_progress_bar = tqdm(test_dataloader, colour="yellow")
        # for i, (images, targets) in enumerate(val_progress_bar):
        #     images = [image.to(device) for image in images]
        #     with torch.no_grad():
        #         predictions = model(images)
        #     for prediction in predictions:
        #         print(prediction["boxes"])
        #         print(prediction["labels"])
        #         print(prediction["scores"])

