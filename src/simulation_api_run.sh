docker run -it --shm-size=100G -p 8080:8080 -d -v $(pwd):$(pwd) --name sewer_api sewer_defect_yolo \
	/bin/bash -c "cd /home/ubuntu/workspace/sewer_api/ && python3 sewer_simulation_api.py"
