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

## Train with Darknet

Log in to Ubuntu DSVM:

- Must first delete an inbound port rule under “Networking” in the Azure Portal (delete Cleanuptool-Deny-103)
- ssh into machine w/ username and password (of if used ssh key, use that)

Get pretrained weights and test detector to ensure program works.
```
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
./darknet detector test ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./yolov4-tiny.weights
```

- Install Yolo_mark (build instructions for Windows, Macos, Linux)
- Label data
- Zip and copy (scp) data up to VM (may need to delete network rule 103 again)
- Unzip the compressed data folder into repo (darknet) folder build/darknet/x64

How to train on your own data:  https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects (update cfg file):

Place data in “data” folder
- train.txt – training image (path to images in train.txt and valid.txt should look like:  build/darknet/x64/data/img/imagename.jpg)
- valid.txt – validation set during training
- obj.data – points to all necessary files
- obj.names – the class names one per line

Calculate anchor boxes (especially important if you have very big or very small objects on average)

`./darknet detector calc_anchors build/darknet/x64/data/obj.data -num_of_clusters 6 -width 416 -height 416`

Configure the cfg file (cfg/yolov4-tiny-custom.cfg):
- Batch size – the number of images feed into DNN as one unit before a weight update happens
- Iterations – number of passes over dataset (not full passes usually, just batch-sized passes); also called epochs
- Learning rate – if decrease this, then increase number of iterations (called max_batches in config file)
- Class number – change to your number of classes (each YOLO block)
- Filters – (5 + num_classes)*3  (each YOLO block)
- Anchors – these are also known as anchor boxes (each YOLO block)

```
 wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
./darknet detector train build/darknet/x64/data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -map -dont_show
```


## TFlite conversion

You can download your model and do the following on your local machine.

Download model:

`scp scp <username>@<public IP or DNS name>:~/darknet/backup/yolov4-tiny-custom_best.weights .`

Clone repo:

`git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git`

You can use an editor like VSCode, now, on your local machine.  Change `coco.names` to be your classes (`obj.name`) in `core/config.py` and place `obj.names` in the `data/classes` folder.

Darknet to TensorFlow Lite (with quantization):

```
python save_model.py --weights yolov4-tiny-custom_best.weights --output ./checkpoints/yolov4-tiny-416-tflite2 --input_size 416 --model yolov4 --framework tflite --tiny
python convert_tflite.py --weights ./checkpoints/yolov4-tiny-416-tflite2 --output ./checkpoints/yolov4-tiny-416-fp16.tflite --quantize_mode float16
```

Run the video test:

`python detectvideo.py --framework tflite --weights ./checkpoints/yolov4-tiny-416-fp16.tflite --size 416 --tiny --model yolov4 --video 0 --score 0.4`

