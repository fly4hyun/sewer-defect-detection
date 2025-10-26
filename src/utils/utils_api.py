import os
import json
import requests
from datetime import datetime, timedelta

import io
from PIL import Image
import base64

class DefectMap:
    def __init__(self, n_classes: int):
        self.defects, self.colors = self._get_defects_and_colors(n_classes)

    def _get_defects_and_colors(self, n_classes: int):
        '''Define main, other defects, and their color maps based on defect type.'''
        # Main defects with colors(RGB)
        main_defects = {
            0: '균열',
            1: '연결관',
            2: '이음부',
            3: '토사퇴적',
            4: '파손',
            5: '표면손상',
        }
        main_colors = {
            0: (180, 180, 180),
            1: (0, 191, 255),
            2: (255, 165, 0),
            3: (50, 205, 50),
            4: (255, 0, 0),
            5: (255, 215, 0),
        }
        
        # Other (etc) defects with colors(RGB)
        other_defects = {
            0: '라이닝결함',
            1: '좌굴',
            2: '변형',
            3: '영구장애물',
            4: '천공',
            5: '침하',
            6: '폐유부착',
            7: '임시장애물',
            8: '뿌리침입',
        }
        other_colors = {
            0: (221, 160, 221),
            1: (30, 144, 255),
            2: (255, 140, 0),
            3: (128, 0, 128),
            4: (255, 20, 147),
            5: (139, 69, 19),
            6: (105, 105, 105),
            7: (255, 105, 180),
            8: (0, 255, 127),
        }

        if n_classes == 6:
            return main_defects, main_colors
        elif n_classes == 9:
            return other_defects, other_colors
        elif n_classes == 15:
            # Combine main and other defects, shifting other defects to avoid overlap
            combined_defects = {**main_defects, **{k + len(main_defects): v for k, v in other_defects.items()}}
            combined_colors = {**main_colors, **{k + len(main_colors): v for k, v in other_colors.items()}}
            return combined_defects, combined_colors
        else:
            return {}, {}

    def get_defect_name(self, defect_id: int) -> str:
        '''Return the name of the defect based on the defect ID.'''
        return self.defects.get(defect_id, 'Unknown Defect')

    def get_defect_color(self, defect_id: int) -> tuple:
        '''Return the BGR color associated with a defect ID.'''
        return self.colors.get(defect_id, (0, 0, 0))  # Default to black if not found

    def all_defects(self):
        '''Return all defects as a dictionary.'''
        return self.defects
    

def convert_to_mapping(cls: int, n_classes: int) -> str:
    '''
    Convert a class index to a category mapping.

    Args:
        cls (int): class index, main defect n_classes is 6, etc n_classes is 9, all n_classes is 15
        n_classes (int): number of classes

    Returns:
        str: category mapping
    '''
    if n_classes == 6:
        return ('주요결함')
    elif n_classes == 9:
        return ('기타결함')
    elif n_classes == 15:
        if 0 <= cls <= 5:
            return('주요결함')
        else:
            return('기타결함')


def image_to_base64(image: Image) -> base64:
    buffered = io.BytesIO()
    image.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def delete_old_files(save_path: str, days_threshold: int) -> None:
    '''
    Delete old files in the specified directory based on the days threshold.

    Args:
        save_path (str): The path of the directory.
        days_threshold (int): The number of days before the files are deleted.
    '''
    # Ensure results directory exists
    os.makedirs(save_path, exist_ok=True)

    current_time = datetime.now()
    time_threshold = timedelta(days=days_threshold)

    file_list = os.listdir(save_path)
    for file_name in file_list:
        file_path = os.path.join(save_path, file_name)

        # Process files only
        if os.path.isfile(file_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            time_difference = current_time - file_mod_time

            if time_difference > time_threshold:
                os.remove(file_path)
                print(f'Deleted: {file_name}')


def send_to_api(output: dict):
    headers = {'Content-Type': 'application/json'}
    response = requests.post('http://13.209.138.73:8080/api/airesearch/result', data=json.dumps(output, indent=4), headers=headers)

    if response.status_code == 200:
        print('Success:', response.text)
    else:
        print('Failed:', response.status_code)


def create_output(results_pipe_info, data, video_url, key, status='F', msg=None):
    """
    Create a dictionary that contains the results of pipe information extraction.

    Args:
        results_pipe_info (dict): A dictionary containing the extracted pipe information.
        data (list): A list of bounding box coordinates and defect information.
        video_url (str): The URL of the video.
        key (str): The identifier for the video.
        status (str, optional): The status of the extraction. Schould be 'S' or 'F'
        msg (str, optional): Print result message

    Returns:
        dict: A dictionary containing the extracted pipe information, bounding box coordinates and defect information, video URL, and the status of the extraction.
    """
    output = {
        'pipe_start': results_pipe_info['pipe_start'],
        'pipe_end': results_pipe_info['pipe_end'],
        'pipe_number': results_pipe_info['pipe_number'],
        'date': results_pipe_info['date'],
        'data': data,
        'video_url': video_url,
        'key': key,
        'status': status,
        'msg': msg
    }
    
    return output


