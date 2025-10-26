###################################################################################################
###################################################################################################

import os
import argparse

from tqdm import tqdm

from ultralytics import YOLO

from utils.utils_postprocess import *
from utils.utils_evaluate import *
from utils.utils_results_image import *
from utils.utils_results_txt import *
from utils.utils_results_visualize import *
from utils.utils_classifier import *
from utils.utils import *
from utils.utils_api import DefectMap

###################################################################################################



###################################################################################################




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
    parser.add_argument('--model', type=str, default='YOLOv10x', help='model name')
    parser.add_argument('--datasets', type=str, default='main_pipe_datasets', help='datasets name')
    parser.add_argument('--datasets_type', type=str, default='test', help='datasets type')
    
    ###
    parser.add_argument('--postprocess', type=bool, default=True)
    parser.add_argument('--postprocess_inter_iom_threshold', type=float, default=0.7)
    parser.add_argument('--postprocess_intra_iom_threshold', type=float, default=1.0)
    parser.add_argument('--classifier', type=bool, default=False)
    
    ###
    parser.add_argument('--save_image', type=bool, default=True)
    parser.add_argument('--save_text', type=bool, default=True)
    parser.add_argument('--save_cm', type=bool, default=True)
    
    ### 
    parser.add_argument('--classifier_model', type=str, default='model_best2.pth', help='Write your classifier model path')
    
    opt = parser.parse_args()
    return opt

###################################################################################################

def main(opt):
    
    num_classes = opt.class_number
    
    model = opt.model
    if model == "YOLOv10x":
        yolo_model = YOLO('weight/YOLOv10x.pt')
        datasets = 'main_pipe_datasets'
    elif model == "YOLOv10x_etc":
        yolo_model = YOLO('weight/YOLOv10x_etc.pt')
        datasets = 'etc_pipe_datasets'
    elif model == "YOLOv10x_all":
        yolo_model = YOLO('weight/YOLOv10x_all.pt')
        datasets = 'all_pipe_datasets_all'
    else:
        yolo_model = YOLO('weight/' + model + '.pt')
        datasets = opt.datasets
    ###
    num_classes = len(yolo_model.names)
    
    class_names = yolo_model.names
    class_names[len(list(class_names.keys()))] = '배경'
    
    cm = torch.zeros((num_classes + 1, num_classes + 1), dtype=torch.int64)
    
    yolo_classification_model = YoloClassifierPostprocess(
        model_path = os.path.join('classifier_weights', opt.classifier_model),
        device = 'cuda',
        n_classes = 4, # 연결관, 이음부, 파손, 표면손상
        seed = 234
    )
    
    defect_map = DefectMap(n_classes=num_classes)
    
    ###
    folder_path = os.path.join('datasets', datasets)
    image_folder_path = os.path.join(folder_path, 'images', opt.datasets_type)
    label_folder_path = os.path.join(folder_path, 'labels', opt.datasets_type)
    
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_folder_path)}
    image_names = [os.path.splitext(f)[0] for f in os.listdir(image_folder_path) if os.path.splitext(f)[0] in {os.path.splitext(l)[0] for l in os.listdir(label_folder_path)}]

    # 폴더가 없으면 생성 (한 번만)
    model_folder = os.path.join('results', os.path.splitext(opt.model)[0])
    model_folder = create_versioned_folder(model_folder)

    with tqdm(total = len(image_names), desc = f"Processing images") as pbar:
        for image_name in image_names:
            
            ### 라벨 읽기
            label_path = os.path.join(label_folder_path, image_name + '.txt')
            label_data = read_detection_file(label_path)
            
            ### 이미지 읽기
            image_path = os.path.join(image_folder_path, image_files[image_name])
            results = yolo_model.predict(image_path, 
                                         verbose = False, 
                                         conf = opt.model_conf)
            
            results_cls = results[0].boxes.cls.clone().detach().to('cpu')
            results_conf = results[0].boxes.conf.clone().detach().to('cpu')
            results_xywhn = results[0].boxes.xywhn.clone().detach().to('cpu')
            
            results_data = {'cls' : results_cls, 'conf' : results_conf, 'xywhn' : results_xywhn}
            
            ###
            if opt.postprocess:
                results_data = postprocess(results_data, opt.postprocess_inter_iom_threshold, opt.postprocess_intra_iom_threshold)
            
            ###
            if opt.classifier:
                results_data['cls'] = classifier_class(results_data['cls'], results_data['xywhn'], image_path, yolo_classification_model)

            if opt.evaluate_mode == 'iou':
                update_confusion_matrix_iou(cm, results_data, label_data, opt.evaluate_iou_threshold)
            else:
                update_confusion_matrix_iom(cm, results_data, label_data, opt.evaluate_iom_threshold)

            # create_output_image 호출 시 image_files 전달
            if opt.save_image:
                create_output_image(image_name, results_data, label_data, image_folder_path, model_folder, image_files, defect_map)
            
            pbar.update(1)
    
    final_overall_metrics, final_metrics = calculate_final_metrics(cm)
    
    if opt.save_cm == True:
        plot_confusion_matrix(cm, class_names, model_folder)
        plot_metrics(final_metrics, final_overall_metrics, class_names, model_folder)
        
    print_confusion_matrix(cm, class_names)
    print_metrics(final_overall_metrics, final_metrics)
    
    if opt.save_text:
        save_results_to_files(cm, class_names, final_overall_metrics, final_metrics, model_folder, defect_map)

###################################################################################################
###################################################################################################

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

###################################################################################################
###################################################################################################