import re
import cv2

def enhance_contrast(image):
    # Grayscale 변환 (컬러 이미지에서만 수행)
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # CLAHE 적용 - clipLimit 조정
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))  # clipLimit 줄임
    enhanced_image = clahe.apply(gray)
    return enhanced_image


def enhance_edges(image):
    # Grayscale 변환
    if len(image.shape) == 3 and image.shape[2] == 3:  # 컬러 이미지
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # 이미 그레이스케일인 경우
        gray = image

    # Canny Edge Detection
    edges = cv2.Canny(gray, 100, 200)
    
    # 원본과 결합
    enhanced_image = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)
    return enhanced_image


def apply_threshold(image):
    # Grayscale 변환
    if len(image.shape) == 3 and image.shape[2] == 3:  # 컬러 이미지
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Adaptive Thresholding 적용
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
    return binary_image


def dilate_image(image):
    # Grayscale 변환
    if len(image.shape) == 3 and image.shape[2] == 3:  # 컬러 이미지
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 이진화
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilation 적용
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    return dilated_image


def correct_ocr_errors(text):
    corrections = {
        'O': '0', 'o': '0', '@': '0', 'D': '0', 'C': '0', 'c' : '0', 'U': '0', '(' :'0', ')': '0', # O나 o를 0으로
        'l': '1', 'I': '1', 'i': '1',   # 소문자 l이나 대문자 I를 1로
        'Z': '2', 'z': '2',           # Z를 2로
        'A': '4',            # A를 4로
        'S': '5',            # S를 5로
        'T': '7',
        'B': '8', '&': '8',            # B를 8로
        'g': '9',
        ' ': '', '_': '', 
        ',': '.'
    }
    corrected_text = ''.join(corrections.get(char, char) for char in text)
    return corrected_text


def correct_distance_ocr_errors(text, now_distance):
    # 1. 기존 거리 형식에서 숫자 부분과 소수점 부분 분리
    def parse_distance(distance):
        if not distance or 'm' not in distance:
            return None, None
        match = re.match(r'^(\d{1,3})(?:\.(\d))?m$', distance)
        if match:
            integer_part = match.group(1).zfill(3)
            decimal_part = match.group(2) or '0'
            return integer_part, decimal_part
        return None, None


    # 2. 입력된 텍스트를 숫자 형식으로 보정
    def fix_text_format(text):
        # 불필요한 문자 제거 및 소수점 처리
        text = re.sub(r'[^0-9.]', '', text)
        
        # 여러 개의 .을 하나로 줄이기
        text = re.sub(r'\.+', '.', text)
        
        # 소수점이 없으면 세 번째 자리에 추가
        if '.' not in text:
            text = text[:3] + '.' + text[3:]
        
        # 소수점 이하 한 자리로 제한
        parts = text.split('.')
        text = parts[0][:3] + '.' + parts[1][:1] + 'm'
        return text


    # 3. 입력된 텍스트가 소수점 없이 3자리 숫자인 경우 처리
    if text.isdigit() and len(text) == 3:
        int_part, decimal_part = parse_distance(now_distance)
        if int_part == text:
            return f'{text}.{decimal_part}m'
        text = f'{text}.0m'

    # 텍스트 보정
    text = fix_text_format(text)
    current_int, current_dec = parse_distance(now_distance)
    new_int, new_dec = parse_distance(text)
    
    if current_int is None or new_int is None:
        return None
    
    # 4. 비교 및 조건 처리 (정수 부분이 1 이상 증가하지 않도록)
    if (new_int > current_int or (new_int == current_int and new_dec >= current_dec)):
        if int(new_int) - int(current_int) > 1:
            return None
        
        # 소수점 이하 숫자가 4 이상 증가하지 않도록 제한
        if int(new_dec) - int(current_dec) >= 6:
            return None

        # 정수 부분이 증가할 때 소수점 이하가 '0'으로 리셋되지 않으면 제한
        if int(new_int) > int(current_int) and int(new_dec) >= 5:
            return None

        return f'{new_int}.{new_dec}m'
    else:
        return None


def is_fixed_date_format(text):
    # OCR 교정 적용
    corrected_text = correct_ocr_errors(text)
    # 고정된 날짜 형식: YYYY-MM-DD
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    return bool(re.match(pattern, corrected_text)), corrected_text


def ocr_date(reader, frame, date_ocr_check, date_attempts, date_counter, now_date):
    height, width, _ = frame.shape

    if not date_ocr_check:
        cropped_frame_date = frame[:height // 12, -width // 4:]
        results_date_ocr = reader.readtext(cropped_frame_date)
        date_attempts += 1
        
        if results_date_ocr:
            result_date = results_date_ocr[0][1]
            is_valid_format, corrected_date = is_fixed_date_format(result_date)
            if is_valid_format:
                date_counter[corrected_date] = date_counter.get(corrected_date, 0) + 1
        
        # 200번 시도 이후로 가장 많이 감지된 날짜를 결정
        if date_attempts >= 300 and not date_ocr_check:
            if date_counter:
                # 가장 많이 감지된 날짜를 최종 날짜로 결정
                now_date = max(date_counter, key=date_counter.get)
            date_ocr_check = True

    return date_ocr_check, now_date, date_attempts, date_counter


def is_fixed_distance_format(text, now_distance):
    # OCR 교정 적용
    corrected_text = correct_ocr_errors(text)
    corrected_text = correct_distance_ocr_errors(corrected_text, now_distance)
    # ###.#m 형식 확인
    pattern = r'^\d{3}\.\d{1}m$'
    if corrected_text == None:
        return False, text
    return bool(re.match(pattern, corrected_text)), corrected_text
            
            
def ocr_distance(reader, frame, now_distance):
    
    height, width, _ = frame.shape
    
    
    # 상단 1/8 부분 잘라내기
    cropped_frame_distance = frame[:height // 12, :width // 4]
    
    ####
    # cropped_frame_distance = enhance_contrast(cropped_frame_distance)
    
    results_distance_ocr = reader.readtext(cropped_frame_distance)
    if results_distance_ocr != []:
        results_distance = results_distance_ocr[0][1]
        distance_check, now_distance_temp = is_fixed_distance_format(results_distance, now_distance)
        if distance_check == True:
            now_distance = now_distance_temp
    
    return now_distance


### pipe info
def extract_label_numbers(text):
    patterns = {
        '상': r'상[\s:]*(\d{4}-\d{3}-\d{1})',
        '하': r'하[\s:]*(\d{4}-\d{3}-\d{1})',
        '관': r'관로?[\s:]*(\d{2}-\d{5})'  # '관'의 형식이 다르다고 가정
    }
    for label, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            number = match.group(1)
            return label, number
    return None, None


def correct_ocr_errors_ko(text):
    # 정규 표현식을 사용하여 숫자, ':', '-', '상', '하', '관'만을 남기고 나머지는 모두 제거
    corrected_text = re.sub(r'[^0-9:\-상하관]', '', text)
    return corrected_text


def ocr_pipe_info(reader, frame, pipe_check, pipe_info, detection_counters):
    height, width, _ = frame.shape
    cropped_frames = [
        frame[height // 12:int(1.9 * height // 12), :width // 4],
        frame[int(1.9 * height // 12):int(2.6 * height // 12), :width // 4],
        frame[int(2.6 * height // 12):int(3.5 * height // 12), :width // 4]
    ]

    labels = ['상', '하', '관']

    # 각 영역별로 OCR 실행 및 결과 처리
    for i, cropped_frame in enumerate(cropped_frames):
        if not pipe_check[i]:  # 이 레이블에 대한 OCR이 완료되지 않았다면 실행
            results = reader.readtext(cropped_frame, allowlist='상하관:-0123456789')
            detection_counters[i]['attempts'] += 1  # OCR 시도 횟수 증가
            for result in results:
                text = result[1]  # 가장 높은 신뢰도의 텍스트 추정
                corrected_text = correct_ocr_errors_ko(text)
                label, number = extract_label_numbers(corrected_text)
                if label and number:
                    idx = labels.index(label)
                    if not pipe_check[idx]:  # 이 레이블에 대해 아직 확정되지 않았다면
                        detection_counters[idx]['numbers'][number] = detection_counters[idx]['numbers'].get(number, 0) + 1

            # 100번 시도했으나 번호를 찾지 못한 경우
            if detection_counters[i]['attempts'] >= 300:
                pipe_check[i] = True
                if not pipe_info[i]:  # 아직 유효한 번호가 없다면
                    if detection_counters[i]['numbers']:  # 검출된 숫자가 하나라도 있는 경우
                        pipe_info[i] = max(detection_counters[i]['numbers'], key=detection_counters[i]['numbers'].get)
                    else:
                        pipe_info[i] = None  # 검출된 숫자가 없는 경우

    return pipe_check, pipe_info







