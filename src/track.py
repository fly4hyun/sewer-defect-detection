###################################################################################################
###################################################################################################


import os
import argparse

from tqdm import tqdm

from ultralytics import YOLO
from collections import defaultdict

import easyocr

from utils.utils_postprocess import *
from utils.utils_evaluate import *
from utils.utils_results_image import *
from utils.utils_results_txt import *
from utils.utils_results_visualize import *
from utils.utils_classifier import *
from utils.utils_ocr import *

from utils.utils_track_postprocess import *
from utils.utils_track_visualize import *
from utils.utils_api import DefectMap, convert_to_mapping

###################################################################################################

def parse_opt():
    parser = argparse.ArgumentParser()
    
    ###
    parser.add_argument('--class_number', type=int, default=8)
    parser.add_argument('--model_conf', type=float, default=0.3)
    parser.add_argument('--evaluate_mode', type=str, default='iou')
    parser.add_argument('--evaluate_iou_threshold', type=float, default=0.5)
    parser.add_argument('--evaluate_iom_threshold', type=float, default=0.5)
    parser.add_argument('--asd', type=int, default=8)
    
    ###
    parser.add_argument('--model', type=str, default='YOLOv10x_all', help='model name')
    parser.add_argument('--video', type=str, default='video/05-06072.mp4', help='datasets name')
    parser.add_argument('--datasets_type', type=str, default='test', help='datasets type')
    
    ###
    parser.add_argument('--postprocess', type=bool, default=True)
    parser.add_argument('--postprocess_inter_iom_threshold', type=float, default=0.7)
    parser.add_argument('--postprocess_intra_iom_threshold', type=float, default=1.0)
    parser.add_argument('--classifier', type=bool, default=True)
    
    ###
    parser.add_argument('--save_image', type=bool, default=True)
    parser.add_argument('--save_text', type=bool, default=True)
    parser.add_argument('--save_cm', type=bool, default=True)
    
    ### 
    parser.add_argument('--classifier_model', type=str, default='model_best2.pth', help='Write your classifier model path')
    
    ###
    parser.add_argument('--save_video', type=bool, default=True)
    parser.add_argument('--save_video_image', type=bool, default=True)

    opt = parser.parse_args()
    return opt

###################################################################################################

def main(opt):
    
    # Initialize OCR readers for English and Korean
    reader = easyocr.Reader(['en'], gpu=True)
    reader_ko = easyocr.Reader(['ko'], gpu=True)

    # Load YOLO model
    yolo_model = YOLO(os.path.join('weight', opt.model + '.pt'))
    class_names = yolo_model.names
    n_classes = len(class_names)
    class_names[len(list(class_names.keys()))] = 'backg'
    defect_map = DefectMap(n_classes=n_classes)
    
    ###
    yolo_classification_model = YoloClassifierPostprocess(
        model_path = os.path.join('classifier_weights', opt.classifier_model),
        device = 'cuda',
        n_classes = 4, # 연결관, 이음부, 파손, 표면손상
        seed = 234
    )
    
    # Load video from URL
    video_path = opt.video
    cap = cv2.VideoCapture(video_path)
    # Setup video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 10
    frame_skip = int(fps / target_fps) if fps > target_fps else 1
    
    # Initialize variables for OCR and tracking
    pipe_check = [False, False, False]
    pipe_info = [None, None, None]
    detection_counters = [{'numbers': {}, 'attempts': 0} for _ in range(3)]
    date_ocr_check = False
    date_attempts = 0
    date_counter = {}
    now_distance = '000.0m'
    now_date = None

    # Video writer setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 동영상 저장을 위한 VideoWriter 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
    output_path = 'video_test_results/' + video_path.split('/')[-1]  # 파일 경로 수정
    os.makedirs("video_test_results/", exist_ok=True)
    if opt.save_video:
        out = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))  # VideoWriter 생성
    
    ### save video results
    name_video = video_path.split('/')[-1]
    name_video_f = name_video.split('.')[0]
    
    os.makedirs("video_test_image_results/" + name_video_f, exist_ok=True)
    
    # Initialize tracking history
    track_history = defaultdict(lambda: [])
    track_frame = defaultdict(lambda: [])
    index_count = 1
    
    # Data container for results
    data = []

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if not success or frame is None or frame.size == 0:
            break
        
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame % frame_skip != 0:
            continue
            
        # Extract frame properties
        height, width, _ = frame.shape
        current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = current_frame_number / fps

        # OCR for pipe info, date, and distance
        pipe_check, pipe_info = ocr_pipe_info(reader_ko, frame, pipe_check, pipe_info, detection_counters)
        date_ocr_check, now_date, date_attempts, date_counter = ocr_date(reader_ko, frame, date_ocr_check, date_attempts, date_counter, now_date)
        now_distance = ocr_distance(reader, frame, now_distance)

        results_pipe_info = {
            'pipe_start': pipe_info[0],
            'pipe_end': pipe_info[1],
            'pipe_number': pipe_info[2],
            'date': now_date
        }
        
        # Perform YOLO tracking if distance is sufficient
        if float(now_distance[:-1]) > 0.7:
            results = yolo_model.track(frame, persist=True, stream=False, verbose=False, conf=opt.model_conf)[0]

            if hasattr(results.boxes, 'id') and results.boxes.id is not None:
                results_data = {
                    'cls': results.boxes.cls.cpu(),
                    'conf': results.boxes.conf.cpu(),
                    'xywhn': results.boxes.xywhn.cpu(),
                    'ids': results.boxes.id.int().cpu().tolist(),
                    'ids_old': results.boxes.id.int().cpu().tolist(),
                    'length': now_distance
                }

                # Postprocess results
                results_data_temp = {key: results_data[key] for key in ['cls', 'conf', 'xywhn']}
                results_data_temp = postprocess(results_data_temp, 0.7, 1.0)
                results_data.update(results_data_temp)

                # Update tracking history
                time_id = str(current_time)
                track_history[time_id] = results_data
                track_frame[time_id] = current_frame_number
                results_data, index_count = track_postprocess(track_history, time_id, track_frame, frame_skip, index_count)
                track_history[time_id] = results_data

                # Clean old tracking data
                for track_time in list(track_history.keys()):
                    if track_history[str(track_time)] != []:
                        if float(now_distance[:-1]) - float(track_history[str(track_time)]['length'][:-1]) > 1.0:
                            del track_history[str(track_time)]
                            del track_frame[str(track_time)]

                # Prepare detection results
                results_cls_list = results_data['cls'].tolist()
                categorie = []
                kind = []

                for cls in results_cls_list:
                    kind.append(defect_map.get_defect_name(int(cls)))
                    categorie.append(convert_to_mapping(int(cls), n_classes))

                num_detections = len(results_cls_list)
                frames = [int(current_frame_number / (fps / target_fps))] * num_detections
                times = [current_time] * num_detections

                results_data['categorie'] = categorie
                results_data['time'] = times
                results_data['frame'] = frames
                results_data['mapping_class'] = kind

                for data_i in range(len(categorie)):
                    data.append([
                        results_data['categorie'][data_i], 
                        results_data['mapping_class'][data_i], 
                        results_data['conf'][data_i].tolist(), 
                        results_data['time'][data_i], 
                        results_data['frame'][data_i], 
                        results_data['length'], 
                        results_data['ids'][data_i]
                    ])
                
                # Annotate frame
                frame = draw_bounding_boxes(frame, results_data, defect_map)

        
        # Write frame to output video
        if opt.save_video_image:
            name_video_path = 'video_test_image_results/' + name_video_f + '/' + str(current_frame_number) + '_' + name_video.replace('mp4','jpg')
            image_path = name_video_path
            cv2.imwrite(image_path, frame)
        
        # Write frame to output video
        if opt.save_video:
            out.write(frame)

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    if opt.save_video:
        out.release()
    cv2.destroyAllWindows()

    # Filter data based on distance threshold
    distance_threshold = float(now_distance[:-1]) - 2.0
    data = [item for item in data if float(item[5][:-1]) < distance_threshold]
    
    ############################
    
    # Prepare output
    output = {
        'pipe_start': results_pipe_info['pipe_start'],
        'pipe_end': results_pipe_info['pipe_end'],
        'pipe_number': results_pipe_info['pipe_number'],
        'date': results_pipe_info['date'],
        'data': data,
        'key': 'key'
    }
    
###################################################################################################

if __name__ == "__main__":
    
    opt = parse_opt()
    main(opt)

###################################################################################################
###################################################################################################