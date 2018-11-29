# A PyTorch implementation of a YOLO v3 Object Detector

This repository contains code for a object detector based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), implementedin PyTorch. The code is based on the official code of [YOLO v3](https://github.com/pjreddie/darknet), as well as a PyTorch 
port of the original code, by [marvis](https://github.com/marvis/pytorch-yolo2). One of the goals of this code is to improve
upon the original port by removing redundant parts of the code (The official code is basically a fully blown deep learning 
library, and includes stuff like sequence models, which are not used in YOLO).


The weights file can be downloaded into your repo directory as follow (if Linux)

```
mkdir weights
cd weights
wget https://pjreddie.com/media/files/yolov3.weights 
cd ..
python sample_detector.py
```

Modified the original code forked from: https://github.com/ayooshkathuria/pytorch-yolo-v3