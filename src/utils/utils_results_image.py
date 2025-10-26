import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

def draw_boxes_on_image(image, boxes, labels, confidences, image_size, defect_map, is_ground_truth=False):
    '''
    이미지 위에 박스와 클래스 라벨, confidence 값을 그립니다.
    :param image: 원본 이미지
    :param boxes: 이미지 위의 박스 좌표들
    :param labels: 각 박스의 클래스 레이블 (tensor -> int 변환 필요)
    :param confidences: 각 박스의 confidence 값
    :param image_size: 이미지 크기에 따라 폰트와 선 굵기 조정
    :param is_ground_truth: 정답 라벨인 경우 True, 예측 결과는 False
    '''
    # 이미지를 RGBA 모드로 변환하여 투명도 처리
    image = image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    # 이미지 크기에 따른 폰트와 선 굵기 설정
    width, height = image_size
    font_size = max(int(height * 0.06), 30)  # 폰트 크기 조정
    line_width = max(int(width * 0.01), 6)   # 선 굵기 조정

    # 한글 폰트 설정 (시스템에 설치된 한글 폰트 경로로 변경)
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        font = ImageFont.load_default()

    for i, (box, label, conf) in enumerate(zip(boxes, labels, confidences)):
        # 박스 좌표와 클래스 이름 및 confidence 값
        x, y, w, h = box
        x1 = max(0, x - w / 2)
        y1 = max(0, y - h / 2)
        x2 = min(width, x + w / 2)
        y2 = min(height, y + h / 2)
        label_mapping, box_color = defect_map.get_defect_name(int(label)), defect_map.get_defect_color(int(label))

        # 박스 그리기 (선 굵게)
        draw.rectangle([x1, y1, x2, y2], outline=box_color + (255,), width=line_width)

        # 클래스 이름과 confidence 값 그리기 (예측인 경우 conf 포함)
        if not is_ground_truth:
            text = f'{int(label)}:{label_mapping} {conf:.2f}'
        else:
            text = label_mapping
            
        # 텍스트 크기 및 배경 크기 설정
        _, _, text_width, text_height = font.getbbox(text)

        # 텍스트가 이미지 바깥으로 나가는 경우 위치 조정
        text_x1 = x1
        text_y1 = y1 - text_height - line_width # 박스 아래쪽에 텍스트 박스가 붙도록

        if text_y1 <= 0:
            text_y1 = y1 + line_width
            text_x1 = x1 + line_width

        # 텍스트 배경을 그려서 가독성 향상 (투명도 적용)
        text_bg = [text_x1, text_y1, text_x1 + text_width, text_y1 + text_height + line_width]
        draw.rectangle(text_bg, outline=box_color + (255,), fill=box_color + (150,))  # 투명도 150 적용

        # 텍스트 그리기 (하얀색 글씨)
        draw.text((text_x1, text_y1), text, font=font, fill=(255, 255, 255, 255))

    # 원본 이미지와 오버레이 합성
    image = Image.alpha_composite(image, overlay)
    # RGBA에서 RGB로 변환하여 반환
    return image.convert('RGB')


def create_output_image(image_name, results_data, label_data, image_folder_path, model_folder, image_files, defect_map):
    # 이미지 파일 불러오기 (확장자를 동적으로 처리)
    image_path = os.path.join(image_folder_path, image_files[image_name])  # 확장자를 포함한 파일 이름 사용
    image = Image.open(image_path).convert('RGB')

    # 이미지 크기
    width, height = image.size

    # 정답 박스 그리기
    gt_boxes = label_data['xywhn'] * np.array([width, height, width, height])
    gt_image = draw_boxes_on_image(image.copy(), gt_boxes, label_data['cls'], [1] * len(gt_boxes), image.size, defect_map, is_ground_truth=True)

    # 예측 결과가 없을 경우 처리
    if len(results_data['xywhn']) > 0:
        pred_boxes = results_data['xywhn'] * np.array([width, height, width, height])
        pred_image = draw_boxes_on_image(image.copy(), pred_boxes, results_data['cls'], results_data['conf'], image.size, defect_map, is_ground_truth=False)
    else:
        pred_image = image.copy()  # 예측 결과가 없으면 원본 이미지를 그대로 사용

    # 이미지를 나란히 붙이기
    combined_image = Image.new('RGB', (width * 2, height))
    combined_image.paste(gt_image, (0, 0))
    combined_image.paste(pred_image, (width, 0))

    # 결과 이미지를 저장할 폴더 생성
    model_folder = os.path.join(model_folder, 'results_image')
    os.makedirs(model_folder, exist_ok=True)

    # 저장할 이미지 이름 설정
    image_filename = os.path.splitext(image_files[image_name])[0] + '.jpg'
    save_path = os.path.join(model_folder, image_filename)

    # 이미지 저장
    combined_image.save(save_path)