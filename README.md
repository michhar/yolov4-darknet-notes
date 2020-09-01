# yolov4-darknet-notes

## Setup on Ubuntu (16.04) DSVM

Update, OpenCV, clone and setup for build:

```
sudo cp /etc/apt/sources.list.d/tensorflow-serving.list /etc/apt/sources.list.d/tensorflow-serving.list.save`
sudo rm /etc/apt/sources.list.d/tensorflow-serving.list
sudo apt update
sudo apt-get install python-opencv
git clone https://github.com/AlexeyAB/darknet.git
cd darknet/
vim Makefile
```

Change's in Makefile:
```
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=1
OPENMP=1
LIBSO=1
```

Build: `make`

Test:
```
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
./darknet detector test ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./yolov4-tiny.weights
```

Check `predictions.jpg` for results.

