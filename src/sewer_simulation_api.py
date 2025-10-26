import os
from datetime import datetime
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image
import io
from typing_extensions import Annotated
from ultralytics import YOLO

from utils.utils_postprocess import postprocess
from utils.utils_results_image import draw_boxes_on_image
from utils.utils_api import convert_to_mapping, image_to_base64, DefectMap

app = FastAPI()

# 출력 데이터 모델 정의
class OutputData(BaseModel):
    image: str
    data: List[List[str]]
    key: str
    time: float
    defect_count: int


@app.post('/simulation')
async def simulation(
    file: Annotated[UploadFile, File()],
    confidence: Annotated[float, Form()],
    model: Annotated[str, Form()],
    key: Annotated[str, Form()]
) -> OutputData:
    """
    This function takes an image file, confidence threshold, model name, and key as input,
    and returns a JSON object containing the image, defect data, key, and time taken.
    
    Args:
        file (UploadFile): The image file to be processed.
        confidence (float): The confidence threshold for the model to detect defects.
        model (str): The name of the model to use for defect detection.
        key (str): The key to be used for authentication.
    
    Returns:
        OutputData: A JSON object containing the image, defect data, key, and time taken.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    start_time = datetime.now()

    if model == 'YOLOv10x':
        infer_model = YOLO('weight/YOLOv10x.pt')
    elif model == 'YOLOv10x_etc':
        infer_model = YOLO('weight/YOLOv10x_etc.pt')
    elif model == 'YOLOv10x_all':
        infer_model = YOLO('weight/YOLOv10x_all.pt')

    n_classes = len(infer_model.names)
    results = infer_model.predict(image, verbose=False, conf=confidence)
    
    results_data = {
        'cls': results[0].boxes.cls.clone().detach().to('cpu'),
        'conf': results[0].boxes.conf.clone().detach().to('cpu'),
        'xywhn': results[0].boxes.xywhn.clone().detach().to('cpu')
    }
    results_data = postprocess(results_data, 0.7, 1.0)  # (results_data, inter iom threshold, intra iom threshold)

    if results_data['cls'].numel() == 0:
        return OutputData(
            image=image_to_base64(image),
            data=[],
            key=key,
            time=0,
            defect_count=0
        )
    
    width, height = image.size
    defect_map = DefectMap(n_classes=n_classes)
    results_data['mapping_class'] = [defect_map.get_defect_name(cls_id.item()) for cls_id in results_data['cls']]
    gt_boxes = results_data['xywhn'] * np.array([width, height, width, height])
    gt_image = draw_boxes_on_image(image.copy(), gt_boxes, results_data['cls'], [1] * len(gt_boxes), image.size, defect_map, is_ground_truth=True)
    gt_image.save('example2.jpg')

    data_list = [[convert_to_mapping(cls_id.item(), n_classes), mapping_class, f'{conf:.4f}'] 
                 for cls_id, mapping_class, conf in zip(results_data['cls'], results_data['mapping_class'], results_data['conf'])]  # [대분류, 결함 유형, conf]
    defect_count = len(data_list)  # 결함 수 계산

    end_time = datetime.now()
    duration = round((end_time - start_time).total_seconds(), 2)

    return OutputData(
        image=image_to_base64(gt_image),
        data=data_list,
        key=key,
        time=duration,
        defect_count=defect_count
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run('sewer_simulation_api:app', host='0.0.0.0', port=8080, reload=True)

