import numpy as np
from collections import defaultdict

import torch

from utils.utils import iom

def postprocess(detection_data, inter_iom, intra_iom):
    cls = detection_data['cls']
    conf = detection_data['conf']
    xywhn = detection_data['xywhn']
    
    class_boxes = defaultdict(list)
    for i, cl in enumerate(cls):
        class_boxes[cl.item()].append((xywhn[i], conf[i], i))  # Append index for identification

    # Merge boxes within the same class if they overlap
    merged_boxes = []
    index_map = {}  # Map original indices to their new merged indices
    for cl, boxes in class_boxes.items():
        merged_boxes_temp = []
        while boxes:    
            current_box, current_conf, idx = boxes.pop(0)
            to_merge = [(current_box, current_conf, idx)]
            boxes_check_bool = False

            rest = []
            for other_box, other_conf, other_idx in boxes:
                if iom(current_box, other_box) > inter_iom:  # Assume iou function exists
                    to_merge.append((other_box, other_conf, other_idx))
                    boxes_check_bool = True
                else:
                    rest.append((other_box, other_conf, other_idx))
            if len(to_merge) > 1:
                merged_indices = [b[2] for b in to_merge]
                x_coords = [x - w/2 for x, y, w, h in [b[0] for b in to_merge]] + [x + w/2 for x, y, w, h in [b[0] for b in to_merge]]
                y_coords = [y - h/2 for x, y, w, h in [b[0] for b in to_merge]] + [y + h/2 for x, y, w, h in [b[0] for b in to_merge]]
                merged_x = (min(x_coords) + max(x_coords)) / 2
                merged_y = (min(y_coords) + max(y_coords)) / 2
                merged_w = max(x_coords) - min(x_coords)
                merged_h = max(y_coords) - min(y_coords)
                max_conf = max([b[1] for b in to_merge])
                for idx in merged_indices:
                    index_map[idx] = len(merged_boxes_temp)
                merged_boxes_temp.append((np.array([merged_x, merged_y, merged_w, merged_h]), max_conf, cl))
            else:
                index_map[idx] = len(merged_boxes_temp)
                merged_boxes_temp.append((current_box, current_conf, cl))
            boxes = rest
            if boxes == [] and boxes_check_bool == True:
                boxes = merged_boxes_temp
                merged_boxes_temp = []
        merged_boxes += merged_boxes_temp
    # Compare across different classes using indices to manage overlaps
    final_indices = set(range(len(merged_boxes)))
    for i in range(len(merged_boxes)):
        if i not in final_indices:
            continue
        box1, conf1, cls1 = merged_boxes[i]
        for j in range(len(merged_boxes)):
            if i != j and j in final_indices:
                box2, conf2, cls2 = merged_boxes[j]
                if cls1 != cls2 and iom(box1, box2) >= intra_iom:
                    if conf1 >= conf2:
                        final_indices.discard(j)
                    elif conf1 < conf2:
                        final_indices.discard(i)
                        break

    final_boxes = [merged_boxes[i] for i in final_indices]
    final_cls = [box[2] for box in final_boxes]
    final_conf = [box[1] for box in final_boxes]
    final_xywhn = [box[0] for box in final_boxes]
    
    # 결과 데이터를 반환하는 부분에서, final_xywhn이 비어 있는지 확인
    if len(final_xywhn) > 0:
        results_data = {
            'cls': torch.tensor(final_cls),
            'conf': torch.tensor(final_conf),
            'xywhn': torch.tensor(np.stack(final_xywhn))  # 정상적인 경우
        }
    else:
        # final_xywhn이 비어 있을 경우 처리
        results_data = {
            'cls': torch.tensor([]),  # 빈 텐서 반환
            'conf': torch.tensor([]),  # 빈 텐서 반환
            'xywhn': torch.tensor([])  # 빈 텐서 반환
        }

    return results_data