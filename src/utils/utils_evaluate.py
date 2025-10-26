import torch

from utils.utils import iom, iou

def update_confusion_matrix_iou(cm, results_data, label_data, iou_threshold):
    pred_cls = results_data['cls'].long()
    true_cls = label_data['cls'].long()
    pred_boxes = results_data['xywhn']
    true_boxes = label_data['xywhn']

    num_classes = cm.size(1) - 1  # 마지막 클래스는 배경
    used = torch.zeros(len(true_cls), dtype=torch.bool)

    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_idx = -1

        # IoU 계산을 통해 가장 높은 매칭을 찾음
        for j, true_box in enumerate(true_boxes):
            if not used[j]:  # 이미 매칭된 박스는 건너뜀
                current_iou = iou(pred_box, true_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_idx = j

        if best_iou >= iou_threshold:
            true_index = true_cls[best_idx]
            pred_index = pred_cls[i]
            cm[true_index, pred_index] += 1  # TP 처리
            used[best_idx] = True
        else:
            # 배경에 대해 잘못된 탐지 -> False Positive
            cm[num_classes, pred_cls[i]] += 1  # 배경으로 잘못 탐지된 경우

    # 처리되지 않은 모든 True 박스를 False Negative로 처리
    for idx, was_used in enumerate(used):
        if not was_used:
            true_index = true_cls[idx]
            cm[true_index, num_classes] += 1  # 배경으로 간주하여 FN 처리

    return cm


def update_confusion_matrix_iom(cm, results_data, label_data, iom_threshold):
    pred_cls = results_data['cls'].long()
    true_cls = label_data['cls'].long()
    pred_boxes = results_data['xywhn']
    true_boxes = label_data['xywhn']

    num_classes = cm.size(1) - 1  # 마지막 클래스는 배경
    used = torch.zeros(len(true_cls), dtype=torch.bool)

    for i, pred_box in enumerate(pred_boxes):
        best_iom = 0
        best_idx = -1

        # IoU 계산을 통해 가장 높은 매칭을 찾음
        for j, true_box in enumerate(true_boxes):
            if not used[j]:  # 이미 매칭된 박스는 건너뜀
                current_iou = iom(pred_box, true_box)
                if current_iou > best_iom:
                    best_iom = current_iou
                    best_idx = j

        if best_iom >= iom_threshold:
            true_index = true_cls[best_idx]
            pred_index = pred_cls[i]
            cm[true_index, pred_index] += 1  # TP 처리
            used[best_idx] = True
        else:
            # 배경에 대해 잘못된 탐지 -> False Positive
            cm[num_classes, pred_cls[i]] += 1  # 배경으로 잘못 탐지된 경우

    # 처리되지 않은 모든 True 박스를 False Negative로 처리
    for idx, was_used in enumerate(used):
        if not was_used:
            true_index = true_cls[idx]
            cm[true_index, num_classes] += 1  # 배경으로 간주하여 FN 처리

    return cm


def calculate_final_metrics(cm):
    # 배경은 마지막 클래스이므로 이를 제외한 클래스들에 대해 성능 계산
    num_classes = cm.size(0) - 1  # 배경 제외한 클래스 개수
    tp = torch.diag(cm)[:num_classes].float()  # 배경 제외한 TP
    fp = cm.sum(dim=1)[:num_classes].float() - tp  # 예측 기준 FP
    fn = cm.sum(dim=0)[:num_classes].float() - tp  # 실제 값 기준 FN

    # 모든 값이 0인 클래스 필터링
    valid_classes = (tp + fp + fn) > 0  # TP, FP, FN이 모두 0인 클래스를 제외

    # 유효한 클래스에 대해 성능 계산
    precision = torch.nan_to_num(tp[valid_classes] / (tp[valid_classes] + fp[valid_classes]), nan=0.0)
    recall = torch.nan_to_num(tp[valid_classes] / (tp[valid_classes] + fn[valid_classes]), nan=0.0)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = torch.nan_to_num(f1, nan=0.0)  # NaN을 0으로 처리

    # 각 클래스에 대한 성능 저장
    class_metrics = {
        'precision': precision.tolist(),  # 각 클래스의 정밀도 리스트
        'recall': recall.tolist(),  # 각 클래스의 재현율 리스트
        'f1_score': f1.tolist()  # 각 클래스의 F1 점수 리스트
    }

    # 최종 평균 계산 (배경 제외)
    precision_mean = precision.mean().item() if precision.numel() > 0 else 0
    recall_mean = recall.mean().item() if recall.numel() > 0 else 0
    f1_mean = f1.mean().item() if f1.numel() > 0 else 0

    overall_metrics = {
        'precision': precision_mean,
        'recall': recall_mean,
        'f1_score': f1_mean
    }

    return overall_metrics, class_metrics


def print_metrics(overall_metrics, class_metrics):
    print("=== Per-Class Metrics ===")
    for i, (p, r, f1) in enumerate(zip(class_metrics['precision'], class_metrics['recall'], class_metrics['f1_score'])):
        print(f"Class {i}:")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall:    {r:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print()

    print("=== Overall Metrics ===")
    print(f"Overall Precision: {overall_metrics['precision']:.4f}")
    print(f"Overall Recall:    {overall_metrics['recall']:.4f}")
    print(f"Overall F1 Score:  {overall_metrics['f1_score']:.4f}")


def print_confusion_matrix(cm, class_names):
    '''
    혼동 행렬을 보기 좋게 출력합니다.
    :param cm: Confusion Matrix (Tensor or Numpy array)
    :param class_names: 클래스 이름 딕셔너리
    '''
    # 딕셔너리에서 값만 추출해서 리스트로 변환
    class_names = [class_names[key] for key in sorted(class_names.keys())]
    class_names += ['backg']
    
    # Tensor일 경우 numpy로 변환
    cm = cm.cpu().numpy() if isinstance(cm, torch.Tensor) else cm  

    # 행과 열에 맞춰 출력
    print("Confusion Matrix:")
    print(' ' * 12 + 'Predicted')
    print(' ' * 6 + ' ' + '  '.join(f'{name[:5]:>7}' for name in class_names))
    print('Actual')

    for i, row in enumerate(cm):
        row_values = '  '.join(f'{val:7d}' for val in row)
        print(f'{class_names[i][:5]:<7} | {row_values}')
