import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 한글 폰트 경로 (실제 시스템에 설치된 폰트 경로로 변경하세요)
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

def draw_bounding_boxes(frame, results_data, defect_map):
    # 이미지 크기 가져오기
    height, width, _ = frame.shape

    # results_data에서 데이터 추출
    cls = results_data['cls']
    conf = results_data['conf']
    xywhn = results_data['xywhn']
    ids = results_data['ids']

    # 텐서를 NumPy 배열로 변환 (필요한 경우)
    if not isinstance(cls, np.ndarray):
        cls = cls.cpu().numpy()
    if not isinstance(conf, np.ndarray):
        conf = conf.cpu().numpy()
    if not isinstance(xywhn, np.ndarray):
        xywhn = xywhn.cpu().numpy()
    # 'ids'는 이미 리스트 형태

    # OpenCV 이미지를 PIL 이미지로 변환
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    # 이미지 크기에 따른 폰트와 선 굵기 설정
    font_size = max(int(height * 0.02), 15)  # 폰트 크기 조정
    line_width = max(int(width * 0.005), 2)  # 선 굵기 조정

    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        font = ImageFont.load_default()

    for i in range(len(cls)):
        class_idx = int(cls[i])
        confidence = float(conf[i])
        x_center_n = xywhn[i][0]
        y_center_n = xywhn[i][1]
        box_width_n = xywhn[i][2]
        box_height_n = xywhn[i][3]

        # 좌표 역정규화
        x_center = x_center_n * width
        y_center = y_center_n * height
        box_width = box_width_n * width
        box_height = box_height_n * height

        # 좌상단 및 우하단 좌표 계산
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # 좌표가 이미지 범위를 벗어나지 않도록 조정
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # 클래스 색상 가져오기 (BGR에서 RGB로 변환)
        color = defect_map.get_defect_color(int(class_idx))
    
        # 박스 그리기 (선 굵기 조정)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=line_width)

        # 라벨 준비 (클래스 번호 추가)
        # class_name = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
        class_name = defect_map.get_defect_name(int(class_idx))
        track_id = ids[i]
        text = f"{class_idx}:{class_name} ID:{track_id} Conf:{confidence:.2f}"

        # 라벨 크기 계산
        _, _, text_width, text_height = font.getbbox(text)
        text_x1 = x1
        text_y1 = y1 - text_height - line_width # 박스 아래쪽에 텍스트 박스가 붙도록

        if text_y1 <= 0:
            text_y1 = y1 + line_width
            text_x1 = x1 + line_width

        # 텍스트 배경을 그려서 가독성 향상 (투명도 적용)
        text_bg = [text_x1, text_y1, text_x1 + text_width, text_y1 + text_height + line_width]
        draw.rectangle(text_bg, outline=color + (255,), fill=color + (100,))  # 투명도 150 적용

        # 텍스트 그리기 (하얀색 글씨)
        draw.text((text_x1, text_y1), text, font=font, fill=(255, 255, 255, 255))

    # PIL 이미지를 OpenCV 이미지로 변환
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return frame
##