import os
import torch

def iou(box1, box2):
    """
    Compute the intersection over union (IOU) of two bounding boxes.
    :param box1: [x_center, y_center, width, height] of the first box
    :param box2: [x_center, y_center, width, height] of the second box
    :return: IOU of the two boxes
    """
    # Calculate intersection area
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    intersection_x_min = max(x1_min, x2_min)
    intersection_y_min = max(y1_min, y2_min)
    intersection_x_max = min(x1_max, x2_max)
    intersection_y_max = min(y1_max, y2_max)

    if intersection_x_max < intersection_x_min or intersection_y_max < intersection_y_min:
        return 0.0  # No intersection

    intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    return intersection_area / (box1_area + box2_area - intersection_area)


def iom(box1, box2):
    # Assuming box1 and box2 are in [x_center, y_center, width, height] format
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2

    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_min_x, inter_min_y = max(x1_min, x2_min), max(y1_min, y2_min)
    inter_max_x, inter_max_y = min(x1_max, x2_max), min(y1_max, y2_max)

    if inter_max_x < inter_min_x or inter_max_y < inter_min_y:
        return 0.0  # No overlap

    inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    return inter_area / min(area1, area2)


def read_detection_file(file_path):
    # 빈 텐서 리스트 초기화
    cls_list = []
    xywhn_list = []
    
    # 파일 읽기
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            cls_list.append(int(parts[0]))  # 클래스
            xywhn_list.append([float(x) for x in parts[1:]])  # 좌표

    # 리스트를 텐서로 변환
    cls_tensor = torch.tensor(cls_list)
    xywhn_tensor = torch.tensor(xywhn_list)
    conf_tensor = torch.ones((len(cls_list),))  # 신뢰도를 1로 설정

    # 결과 딕셔너리 생성
    results_data = {
        'cls': cls_tensor,
        'conf': conf_tensor,
        'xywhn': xywhn_tensor
    }

    return results_data


def create_versioned_folder(base_folder):
    """
    폴더가 이미 존재하면 가장 높은 버전 번호를 찾아 +1을 한 후 새 폴더를 생성.
    중간에 비어있는 버전이 있더라도 가장 높은 버전에 +1을 적용.
    :param base_folder: 기본 폴더 경로 (ex. 'results/yolov3')
    :return: 생성된 폴더 경로
    """
    existing_versions = []

    # 기존에 있는 버전 폴더들을 확인
    base_dir = os.path.dirname(base_folder)
    base_name = os.path.basename(base_folder)

    if os.path.exists(base_dir):
        for folder in os.listdir(base_dir):
            if folder.startswith(base_name + '_v'):
                try:
                    # 폴더 이름에서 버전 번호를 추출
                    folder_version = int(folder.split('_v')[-1])
                    existing_versions.append(folder_version)
                except ValueError:
                    pass

    # 가장 높은 버전 번호에 +1 적용
    max_version = max(existing_versions, default=0)
    new_version = max_version + 1
    folder = f"{base_folder}_v{new_version}"

    # 새로운 폴더 생성
    os.makedirs(folder, exist_ok=True)
    return folder