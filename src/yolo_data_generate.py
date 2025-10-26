###################################################################################################
###################################################################################################

import os
import argparse
import shutil
from tqdm import tqdm
from utils.utils_datasets import read_detection_file_xml  # 가정: read_detection_file_xml 함수를 utils에서 가져옴

###################################################################################################
###################################################################################################

def parse_opt():
    parser = argparse.ArgumentParser()
    
    ###
    #parser.add_argument('--data_type', type=str, default='train,val,test', help='train,val,test 쉼표로 구분')
    parser.add_argument('--data_type', type=str, default='val,test', help='train,val,test 쉼표로 구분')
    parser.add_argument('--data_name', type=str, default='서울디지털재단하수관로데이터(전체)')
    parser.add_argument('--data_image_path', type=str, default='data/서울디지털재단하수관로데이터/1.식별화된_결과물(비공개용)/1.B.Box_원본')
    parser.add_argument('--data_label_path', type=str, default='data/서울디지털재단하수관로데이터/1.식별화된_결과물(비공개용)/2.B.Box_메타데이터')
    parser.add_argument('--data_split_txt_path', type=str, default='data')
    
    opt = parser.parse_args()
    
    opt.data_type = opt.data_type.split(",")
    
    return opt

###################################################################################################

def main(opt):
    
    # 사용할 데이터셋 타입을 변수로 설정 (train, val, test)
    dataset_type_list = opt.data_type

    for dataset_type in dataset_type_list:
        print(dataset_type, ' 진행중')
        # 폴더 경로 설정
        data_folder = opt.data_image_path
        all_label_folder = opt.data_label_path
        
        # 데이터셋 이름
        datasets_name = opt.data_name
        datasets_split_txt = opt.data_split_txt_path

        # 새로운 이미지와 라벨 저장 폴더 경로 설정
        images_output_folder = f'./datasets/{datasets_name}/images/{dataset_type}'
        labels_output_folder = f'./datasets/{datasets_name}/labels/{dataset_type}'
        dataset_txt_path = f'./{datasets_split_txt}/{datasets_name}_{dataset_type}.txt'
        save_dataset_txt_path = f'./datasets/{datasets_name}/{dataset_type}.txt'

        # 처리할 이미지 확장자 목록
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

        # 폴더가 존재하지 않으면 생성
        os.makedirs(images_output_folder, exist_ok=True)
        os.makedirs(labels_output_folder, exist_ok=True)

        # all_{dataset_type}.txt 파일에서 파일 이름(확장자 제외)을 읽어옴
        with open(dataset_txt_path, 'r', encoding='utf-8') as f:
            allowed_filenames = set(os.path.splitext(line.strip())[0] for line in f)

        # dataset_type.txt 파일을 쓰기 모드로 열기
        with open(save_dataset_txt_path, 'w', encoding='utf-8') as save_dataset_txt:
            # data_folder 폴더 내 대분류 폴더들 탐색
            for category in tqdm(os.listdir(data_folder), desc="대분류 폴더 탐색"):
                data_category_path = os.path.join(data_folder, category)
                all_label_category_path = os.path.join(all_label_folder, category)

                # 대분류 폴더가 실제로 폴더인지 확인
                if os.path.isdir(data_category_path):
                    # 해당 대분류 폴더 내 파일들을 탐색 (이미지 파일만 처리)
                    for image_file in os.listdir(data_category_path):
                        # 파일이 이미지 확장자인지 확인
                        if any(image_file.lower().endswith(ext) for ext in image_extensions):
                            # 이미지 파일 이름에서 확장자를 제거한 파일명 얻기
                            image_name_without_ext = os.path.splitext(image_file)[0]

                            # 만약 이미지 파일 이름이 allowed_filenames에 없다면 건너뛰기
                            if image_name_without_ext not in allowed_filenames:
                                continue

                            # 이미지 파일 경로와 XML 파일 경로 설정
                            image_path = os.path.join(data_category_path, image_file)
                            xml_file = image_name_without_ext + ".xml"  # 확장자를 제외한 파일 이름에 .xml 추가
                            xml_file_path = os.path.join(all_label_category_path, xml_file)

                            # XML 파일로부터 클래스 및 xywhn 정보 읽기
                            if os.path.exists(xml_file_path):
                                detection_data = read_detection_file_xml(xml_file_path, datasets_name)
                                if detection_data == 'None':
                                    continue

                                # 이미지 파일을 images/{dataset_type} 폴더에 복사
                                new_image_path = os.path.join(images_output_folder, image_file)
                                shutil.copy(image_path, new_image_path)

                                # 라벨 파일을 labels/{dataset_type} 폴더에 저장
                                label_file = image_name_without_ext + ".txt"  # 확장자를 제외한 파일 이름에 .txt 추가
                                label_file_path = os.path.join(labels_output_folder, label_file)
                                with open(label_file_path, 'w', encoding='utf-8') as label_txt:
                                    for cls, xywhn in zip(detection_data['cls'], detection_data['xywhn']):
                                        cls_num = int(cls.item())  # 클래스 번호
                                        x, y, w, h = xywhn.tolist()  # 좌표
                                        label_txt.write(f"{cls_num} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

                                # dataset_type.txt 파일에 이미지 경로 기록
                                save_dataset_txt.write(f"./images/{dataset_type}/{image_file}\n")
                            else:
                                print(f"XML 파일이 없습니다: {xml_file_path}")

    print("모든 파일 처리가 완료되었습니다.")

###################################################################################################
###################################################################################################

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

###################################################################################################
###################################################################################################