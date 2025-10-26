# Standard library imports
import os
import warnings

# Third-party library imports
import numpy as np
import cv2

import torch
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

# Suppress warnings
warnings.filterwarnings(action='ignore')

class Classifier(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        
        self.conv_model = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_384', pretrained=False)
        self.data_config = timm.data.resolve_model_data_config(self.conv_model)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1000, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1) 
            )

    def forward(self, x):
        # print(x.shape) ## torch.Size([batch, 3, 384, 384])
        x = self.conv_model(x)
        # print(x)
        x = self.fc(x)
        return x

class class_mapping():
    def __init__(self):
        self.class_map = {
            0: 1, # 0번 연결관
            1: 2, # 1번 이음부
            2: 4, # 2번 파손
            3: 5  # 3번 표면손상
        }
    
    def __call__(self, x):
        return self.class_map[x]
        

class YoloClassifierPostprocess():
    def __init__(self, model_path, n_classes=4, device='cpu', seed=234):
        self.set_seed(seed, device)
        self.model = Classifier(n_classes=n_classes)
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.class_mapping = class_mapping()
        self.transforms = A.Compose([
            A.Resize(height=384, width=384),  # Resize
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize
            ToTensorV2()  # PyTorch 텐서로 변환
        ])
        self.device = device

    def predict(self, image_path, xywhn, save_result=False):
        if isinstance(image_path, str):
            # image_input is a path
            image = cv2.imread(image_path)
        else:
            # image_input is assumed to be a video frame (numpy array)
            image = image_path
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_height, image_width = image.shape[:2]
        x_center, y_center, width, height = xywhn
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)

        cropped_image = image[y1:y2, x1:x2]

        if save_result:
            image_name = os.path.split(image_path)[-1]
            self.save_cropped_image(cropped_image, image_name)
        
        image_tensor = self.transforms(image=cropped_image)['image']
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)

        preds = output.max(1)[1]
        output = self.class_mapping(preds.item())

        return output

    def set_seed(self, seed, device):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.startswith('cuda'):
            torch.cuda.manual_seed(seed)

    def save_cropped_image(self, cropped_image, output_path):
        # 크롭된 이미지를 지정된 경로에 저장
        os.makedirs('crop_image', exist_ok=True)
        cv2.imwrite(os.path.join('crop_image', output_path), cropped_image)


def classifier_class(results_data_class, results_data_xywhn, image_path, yolo_classification_model):
    '''
    Updates the class indices in results_data_class using a given YOLO classification model.

    Args:
        results_data_class (list): List of class indices.
        results_data_xywhn (list): List of bounding box coordinates (x_center, y_center, width, height).
        image_path (str): Path to the image file.
        yolo_classification_model (YoloClassifierPostprocess): Model for predicting class indices.

    Returns:
        list: Updated list of class indices.
    '''
    for class_idx, each_class in enumerate(results_data_class):
        # Check if the current class is in the target categories for reclassification
        if int(each_class) in [1, 2, 4, 5]:
            # Update the class index using the YOLO classification model's prediction
            results_data_class[class_idx] = yolo_classification_model.predict(image_path, results_data_xywhn[class_idx])

    return results_data_class








