import torch

from utils.utils import iom

def track_postprocess(track_history, time_id, track_frame, frame_skip, index_count):
    
    now_results = track_history[time_id]
    
    cls_list = now_results['cls'].tolist()
    conf_list = now_results['conf'].tolist()
    xywhn_list = now_results['xywhn'].tolist()
    ids_list = now_results['ids']
    ids_old_list = now_results['ids_old']
    now_length = now_results['length']
    
    ## 이음부 통일
    
    class_2_list = []
    for i, cls in enumerate(cls_list):
        if cls == 2:
            class_2_list.append(i)
    
    if len(class_2_list) >= 2:
        selected_2_boxes = [xywhn_list[i] for i in class_2_list]
        
        ## 박스 통일
        min_x = min(box[0] - box[2] / 2 for box in selected_2_boxes)
        max_x = max(box[0] + box[2] / 2 for box in selected_2_boxes)
        min_y = min(box[1] - box[3] / 2 for box in selected_2_boxes)
        max_y = max(box[1] + box[3] / 2 for box in selected_2_boxes)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y
        
        max_2_conf = max(conf_list[i] for i in class_2_list)
        min_2_id = min(ids_list[i] for i in class_2_list)
        
        # 'length' 키는 제외하고 다른 키에 해당하는 리스트에서 값 제거
        for i in sorted(class_2_list, reverse=True):
            del cls_list[i]
            del conf_list[i]
            del xywhn_list[i]
            del ids_list[i]
        
        cls_list.append(2)
        conf_list.append(max_2_conf)
        xywhn_list.append([center_x, center_y, width, height])
        ids_list.append(min_2_id)
    
    ## 새로운 id 추가 여부 확인
    
    new_id_check = [True] * len(cls_list)

    ## 10프레임 이하 중에 겹치는 결함 통일
    now_frame = track_frame[time_id]
    
    for i, cls in enumerate(cls_list):
        overlap_check = False
        for track_time in list(track_history.keys())[::-1]:
            if overlap_check == True:
                break
            if track_time == time_id:
                continue
            if track_frame[str(track_time)] != []:
                if track_frame[str(track_time)] >= now_frame - frame_skip * 5:
                    old_results = track_history[str(track_time)]
                    old_cls_list = old_results['cls'].tolist()
                    old_conf_list = old_results['conf'].tolist()
                    old_xywhn_list = old_results['xywhn'].tolist()
                    old_ids_list = old_results['ids']
                    for j, old_cls in enumerate(old_cls_list):
                        if overlap_check == True:
                            break
                        ## 이전 박스랑 iom 0.3 이상 겹침
                        if iom(xywhn_list[i], old_xywhn_list[j]) > 0.3:
                            overlap_check = True
                            new_id_check[i] = False
                            
                            ## 클래스가 같음
                            if cls == old_cls:
                                ids_list[i] = old_ids_list[j]
                            ## 클래스가 다름
                            else:
                                ids_list[i] = old_ids_list[j]
                                ## 나중에 프레임 5개 ~ 10개 수집해서 정확성 증가시켜야함
                                if conf_list[i] < old_conf_list[j]:
                                    conf_list[i] = old_conf_list[j]
                                    cls_list[i] = old_cls_list[j]

    ## 거리가 같은 경우 + 0.5m 오차까지 허용
    for i, cls in enumerate(cls_list):
        break_check = False
        for track_time in list(track_history.keys())[::-1]:
            if break_check == True:
                break
            if track_time == time_id:
                continue
            if track_history[str(track_time)] != []:
                if float(track_history[str(track_time)]['length'][:-1]) >= float(now_length[:-1]) - 0.5:
                    
                    ### 기존 yolo id랑 같으면 같은 id
                    # for j, old_cls in enumerate(track_history[str(track_time)]['cls'].tolist()):
                    #     if ids_old_list[i] == track_history[str(track_time)]['ids_old'][j]:
                    #         ids_list[i] = track_history[str(track_time)]['ids'][j]
                    #         break_check = True
                    #         new_id_check[i] = False
                    
                    ### 클래스가 같은게 하나 있으면 이전 id 가져감
                    is_unique = cls_list.count(cls) == 1
                    if is_unique:
                        for j, old_cls in enumerate(track_history[str(track_time)]['cls'].tolist()):
                            if cls == old_cls:
                                ids_list[i] = track_history[str(track_time)]['ids'][j]
                                break_check = True
                                new_id_check[i] = False
                                break       

    ## id 재설정
    for i, id_check in enumerate(new_id_check):
        if id_check:
            ids_list[i] = index_count
            index_count = index_count + 1
        
    results_cls = torch.tensor(cls_list)
    results_conf = torch.tensor(conf_list)
    results_xywhn = torch.tensor(xywhn_list)
    results_track_ids = ids_list
    results_track_ids_old = ids_old_list
    
    new_results = {'cls' : results_cls, 
                   'conf' : results_conf, 
                   'xywhn' : results_xywhn, 
                   'ids' : results_track_ids, 
                   'ids_old' : results_track_ids_old, 
                   'length' : now_length}
        
    
    return new_results, index_count