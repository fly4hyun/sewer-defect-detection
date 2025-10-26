###################################################################################################

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import torch
import xml.etree.ElementTree as ET

###################################################################################################

mapping_label_all = {
    '균열(길이)' : 0, 
    'CL' : 0, 
    '균열(복합)' : 0, 
    '균열(원주)' : 0, 
    'Cracks, Vertical (manhole)' : 0, 
    'CC' : 0, 
    'Cracks, Horizontal (manhole)' : 0, 
    'CM' : 0, 
    
    '연결관(돌출)' : 1, 
    'LP' : 1, 
    '연결관(접합부이상)' : 1, 
    'Lateral, Protruding': 1, 
    'ls':1, 
    'LS':1, 
    
    '이음부(단차)' : 2, 
    'JD' : 2, 
    '이음부(손상)' : 2, 
    'JF' : 2, 
    '이음부(이탈)' : 2, 
    '이임부(손상)':2, 
    'Joint, Faulty':2, 
    'Joint, Separated':2, 
    'JS' : 2, 
    
    '토사퇴적' : 3, 
    'DS' : 3, 
    'Deposits, Silty':3, 
    
    '파손' : 4, 
    'BK' : 4, 
    'Broken Pipe' : 4, 
    
    '표면손상' : 5, 
    'SD' : 5,
    'Surface Damage' : 5, 
    
    '라이닝결함' : 6, 
    'LD' : 6, 
    
    'BC' : 7, 
    
    '변형' : 8, 
    'DF' : 8, 
    
    '영구장애물' : 9, 
    'PO' : 9, 
    
    '천공' : 10, 
    'HL' : 10, 
    
    '침하' : 11, 
    'SG' : 11,
    
    '폐유부착' : 12, 
    'DG' : 12,
    
    '임시장애물' : 13, 
    'TO' : 13,
    
    '뿌리침입' : 14, 
    'RI' : 14,
    
}

###################################################################################################

mapping_label_etc = {
    '라이닝결함' : 0, 
    'LD' : 0, 
    
    'BC' : 1, 
    
    '변형' : 2, 
    'DF' : 2, 
    
    '영구장애물' : 3, 
    'PO' : 3, 
    
    '천공' : 4, 
    'HL' : 4, 
    
    '침하' : 5, 
    'SG' : 5,
    
    '폐유부착' : 6, 
    'DG' : 6,
    
    '임시장애물' : 7, 
    'TO' : 7,
    
    '뿌리침입' : 8, 
    'RI' : 8,
    
}

###################################################################################################

mapping_label_main = {
    '균열(길이)' : 0, 
    'CL' : 0, 
    '균열(복합)' : 0, 
    '균열(원주)' : 0, 
    'Cracks, Vertical (manhole)' : 0, 
    'CC' : 0, 
    'Cracks, Horizontal (manhole)' : 0, 
    'CM' : 0, 
    
    '연결관(돌출)' : 1, 
    'LP' : 1, 
    '연결관(접합부이상)' : 1, 
    'Lateral, Protruding': 1, 
    'ls':1, 
    'LS':1, 
    
    '이음부(단차)' : 2, 
    'JD' : 2, 
    '이음부(손상)' : 2, 
    'JF' : 2, 
    '이음부(이탈)' : 2, 
    '이임부(손상)':2, 
    'Joint, Faulty':2, 
    'Joint, Separated':2, 
    'JS' : 2, 
    
    '토사퇴적' : 3, 
    'DS' : 3, 
    'Deposits, Silty':3, 
    
    '파손' : 4, 
    'BK' : 4, 
    'Broken Pipe' : 4, 
    
    '표면손상' : 5, 
    'SD' : 5,
    'Surface Damage' : 5
}

###################################################################################################

def read_detection_file_xml(xml_file, datasets_name):
    
    if datasets_name == '서울디지털재단하수관로데이터(전체)':
        mapping_label = mapping_label_all
        name_except = ['', '내피생성', '침입수', '붕괴', 'IF']
    if datasets_name == '서울디지털재단하수관로데이터(기타)':
        mapping_label = mapping_label_etc
        name_except = ['연결관(접합부이상)', '침입수', '이음부(이탈)', 'LP', 'SD']
    if datasets_name == '서울디지털재단하수관로데이터(대표)':
        mapping_label = mapping_label_main
        name_except = ['', '내피생성', '임시장애물', '천공', '붕괴', '영구장애물', 'IF']

    tree = ET.parse(xml_file)
    root = tree.getroot()

    cls_list = []
    conf_list = []
    xywhn_list = []
    
    # object 태그가 없으면 바로 'None' 반환
    if not root.findall('object'):
        return 'None'

    for obj in root.findall('object'):
        # 필수 태그가 없으면 'None' 반환
        if obj.find('name') is None or obj.find('bndbox') is None:
            return 'None'
        
        bbox = obj.find('bndbox')
        
        if bbox.find('xmin') is None or bbox.find('ymin') is None or bbox.find('xmax') is None or bbox.find('ymax') is None:
            return 'None'

        size_element = root.find('size')
        if size_element is None or size_element.find('width') is None or size_element.find('height') is None:
            return 'None'

        # 정상적인 데이터가 존재할 경우 처리
        name = obj.find('name').text
        if name not in list(mapping_label.keys()):
            return 'None'
        if name in name_except:
            return 'None'
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        width = float(size_element.find('width').text)
        height = float(size_element.find('height').text)
        
        if width == 0 or height == 0:
            return 'None'

        # 박스 중심 좌표 (xy), 너비와 높이 (wh)
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        box_width = xmax - xmin
        box_height = ymax - ymin

        # Normalized 좌표 (xywhn)
        x_center /= width
        y_center /= height
        box_width /= width
        box_height /= height

        # 각 리스트에 데이터 추가
        cls_list.append(mapping_label[name])
        conf_list.append(1.0)  # conf는 1로 고정
        xywhn_list.append([x_center, y_center, box_width, box_height])

    # 최종적으로 하나의 딕셔너리로 반환
    result_dict = {
        'cls': torch.tensor(cls_list),
        'conf': torch.tensor(conf_list),
        'xywhn': torch.tensor(xywhn_list)
    }
    
    return result_dict

###################################################################################################