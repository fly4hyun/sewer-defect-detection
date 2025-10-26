docker run -it --gpus all --shm-size=100G -p 5000:5000 -d -v $(pwd):$(pwd) --name track_sewer_api sewer_defect_yolo \
	/bin/bash "cd /home/ubuntu/workspace/sewer_api/ && python3 track_api_new.py"

 
