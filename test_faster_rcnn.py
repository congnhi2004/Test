import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, \
    FasterRCNN_ResNet50_FPN_V2_Weights
import argparse
from PIL import Image
import numpy as np
import cv2

def get_args():
    parser = argparse.ArgumentParser(description="Test a Animal CNN model")
    parser.add_argument("--model", "-m", type=str, default="trained_models/fasterrcnn_last.pt")
    parser.add_argument("--image", "-i", type=str, default="test_images/9.jpg")
    parser.add_argument("--size", "-s", type=int, default=416)
    parser.add_argument("--threshold", "-t", type=float, default=0.7)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=20 + 1, bias=True)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=4 * (20 + 1), bias=True)
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # image = Image.open(args.image).convert('RGB')
    # image = np.asarray(image)
    ori_image = cv2.imread(args.image)   # BGR
    height, width, _ = ori_image.shape
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.size, args.size)) / 255
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    image = np.transpose(image, (2, 0, 1))
    image = image[None, :, :, :]
    image = torch.from_numpy(image).float()
    image = image.to(device)
    with torch.no_grad():
        predictions = model(image)
    prediction = predictions[0]
    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]
    final_boxes = []
    final_labels = []
    final_scores = []
    for b, l, s in zip(boxes, labels, scores):
        if s > args.threshold:
            final_boxes.append(b)
            final_labels.append(l)
            final_scores.append(s)
    for b, l, s in zip(final_boxes, final_labels, final_scores):
        xmin, ymin, xmax, ymax = b
        xmin = int(xmin / args.size * width)
        xmax = int(xmax / args.size * width)
        ymin = int(ymin / args.size * height)
        ymax = int(ymax / args.size * height)
        ori_image = cv2.rectangle(ori_image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
        ori_image = cv2.putText(ori_image, classes[l] + " %.2f" % s, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (128, 0, 128), 1, cv2.LINE_AA)
    cv2.imwrite("output.jpg", ori_image)




