import os
import time
from pathlib import Path
from collections import defaultdict

import cv2
from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import boto3
import easyocr

from utils.utils_postprocess import postprocess
from utils.utils_ocr import ocr_pipe_info, ocr_date, ocr_distance
from utils.utils_track_postprocess import track_postprocess
from utils.utils_track_visualize import draw_bounding_boxes
from utils.utils_error import handle_error
from utils.utils_api import DefectMap, convert_to_mapping, delete_old_files, send_to_api, create_output

aws_access_key_id = 'AWS_ACCESS_KEY_ID'
aws_secret_access_key = 'AWS_SECRET_ACCESS_KEY'
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
s3_bucket_name = 'BUCKET_NAME'

app = FastAPI()

def upload_to_s3(output_path: str, object_name: str = None) -> str:
    """
    Upload file to S3.

    Args:
        output_path (str): Path to the file to upload.
        object_name (str, optional): Name of the object in S3 to store the file as. Defaults to None.

    Returns:
        str: URL of the object in S3.
    """

    if object_name == None:
        object_name = Path(output_path).name

    result_path = Path(output_path).parent / f'{Path(object_name).stem}_h264.mp4'
    object_name = Path(result_path).name
    # codec 변경 mp4v -> h264
    os.system(f"sudo ffmpeg -y -i {output_path} -c:v libx264 -b:v 1M -speed 5 -c:a libopus -threads 4 {result_path}")

    s3.upload_file(str(result_path), s3_bucket_name, f'sdf_results/{object_name}')
    return f'https://d2n5u3fvxvufx2.cloudfront.net/sdf_results/{object_name}'


async def analyze_video(
        key: str,
        model: str, 
        video_url: str, 
        confidence: float
    ):
    """
    Analyze a video using YOLO model for defect detection and OCR for pipe information.

    Args:
        key (str): Identifier for the video.
        model (str): YOLO model name.
        video_url (str): URL of the video.
        confidence (float): Confidence threshold for detections.
    
    Returns:
        None
    """
    print(f"Start analyzing video {key}")
    print(f"Video URL: {video_url}")
    print(f"Model: {model}")
    print(f"Confidence: {confidence}")
    # Load video from URL
    cap = cv2.VideoCapture(video_url)
    print("Read Video")
    new_filename = video_url.split('/')[-1]
    extension = new_filename.split('.')[-1]
    
    # Initialize OCR readers for English and Korean
    reader = easyocr.Reader(['en'], gpu=True)
    reader_ko = easyocr.Reader(['ko'], gpu=True)
    
    # Load YOLO model
    yolo_model = YOLO(Path('weight') / (model + '.pt'))
    class_names = yolo_model.names
    n_classes = len(class_names)
    class_names[len(list(class_names.keys()))] = 'backg'
    defect_map = DefectMap(n_classes=n_classes)

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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = (key + '.' + extension)
    output_path = Path('results_video') / output_filename
    out = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))

    # Initialize tracking history
    track_history = defaultdict(lambda: [])
    track_frame = defaultdict(lambda: [])
    index_count = 1
    
    # Data container for results
    data = []

    print("Start Video Processing")
    video_t1 = time.time()
    # Process video frames
    while cap.isOpened():
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
            results = yolo_model.track(frame, persist=True, stream=False, verbose=False, conf=confidence)[0]

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
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video processing time: {time.time() - video_t1}")
    print("End Video Processing")
    
    # Filter data based on distance threshold
    distance_threshold = float(now_distance[:-1]) - 2.0
    data = [item for item in data if float(item[5][:-1]) < distance_threshold]

    video_url_t1 = time.time()
    # Upload result video to S3
    print("Start Video Upload")
    video_url = upload_to_s3(output_path)
    print(f"Video upload time: {time.time() - video_url_t1}")
    print("End Video Upload")

    # Prepare output JSON
    output = create_output(results_pipe_info, data, video_url, key, status='S', msg='Success the video')
    
    # Send results to API
    send_to_api(output)


@app.post("/input_video")
async def input_video(
    key: str = Form(...),
    model: str = Form(...),
    video_url: str = Form(...),
    confidence: float = Form(0.3),
    background_tasks: BackgroundTasks = BackgroundTasks()
    ):
    """
    Analyze video and return the results.

    Args:
        key (str): video key
        model (str): model name (YOLOv10x, YOLOv10x_etc, YOLOv10x_all)
        video_url (str): video url(s3 url)
        confidence (float): confidence threshold
        background_tasks (BackgroundTasks): background tasks

    Returns:
        JSONResponse: results of video analysis
    """
    background_tasks.add_task(analyze_video, key, model, video_url, confidence)

    output = {
        'key': key,
        'status': "Success",
        'msg': None,
    }

    # delete old files in the results_video directory after 2 days
    delete_old_files("./results_video", days_threshold=2)

    return JSONResponse(content=output, status_code=200)

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app
    uvicorn.run("track_api_new:app", host="0.0.0.0", port=5000, reload=True)
