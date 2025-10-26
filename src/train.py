###################################################################################################
###################################################################################################

import os
import argparse

from ultralytics import YOLO

###################################################################################################

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='yolov10x', help='model name')
    parser.add_argument('--datasets', type=str, default='main_pipe_datasets', help='datasets name')

    opt = parser.parse_args()
    return opt

###################################################################################################

def main(opt):

    model = YOLO(opt.model + '.pt')
    data_yaml_path = 'data/' + opt.datasets + '.yaml'

    results = model.train(data=data_yaml_path, epochs=800, imgsz=640, batch = 8, name = opt.model)

###################################################################################################

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    
###################################################################################################
###################################################################################################