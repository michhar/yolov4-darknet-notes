# Training a YOLO v4 Darknet Model on Azure and Converting to TFLite for Efficient Inferencing

## Prerequisites

- SSH client or command line tool
- Azure Subscription
- Python 3 installed locally
- Familiarity with Unix commands (`vim`, `nano`, etc.)

## Setup on Ubuntu (16.04) DSVM

1. Set up an N-series DSVM with Darknet by following <a href="https://github.com/michhar/darknet-azure-vm" target="_blank">these instructions</a>. (ensure authentication with username and password).
2. SSH in to Ubuntu DSVM:
    - Must first delete an inbound port rule under “Networking” in the Azure Portal (delete Cleanuptool-Deny-103)
    - SSH into machine w/ username and password (of if used ssh key, use that)
3. Test Darknet by running the following on the command line:
```
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

./darknet detector test ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./yolov4-tiny.weights
```

Check `predictions.jpg` for results.  You may SCP this file down to your machine to view it or alternatively remote desktop into the machine with a program like X2Go.

## Train with Darknet

1. Label some test data locally (aim for 500-1000 boxes drawn, noting that less will result is less accurate results)
    - Install <a href="https://github.com/microsoft/VoTT" target="_blank">VoTT</a> labeling tool to your local/dev machine (there should be instructions and executables for Windows, Macos, Linux)
    - Label data and export as `json`
    - Convert the `json` files to YOLO `.txt` files with a <a href="https://github.com/michhar/azure-and-ml-utils/blob/master/label_tools/vott2.0_to_yolo.py" target="_blank">python script found here</a> (run `python vott2.0_to_yolo.py --help` for usage) which should result in a `.txt` file per `.json`.  The `.txt` files are the YOLO format that `darknet` can use.
    - Structure the folder as follows.
    ```
    data/
        img/
            image1.jpg
            image1.txt
            image2.jpg
            image2.txt
            ...
        train.txt
        valid.txt
        obj.data
        obj.names
    ```
    - Where `obj.data`, a general file to direct `darknet` to the other data-related files and model folder, looks simliar to the following with necessary changes to `classes` for your scenario.
    ```
    classes = 2
    train  = build/darknet/x64/data/train.txt
    valid  = build/darknet/x64/data/valid.txt
    names = build/darknet/x64/data/obj.names
    backup = backup/
    ```
    - Where `obj.names` contains the class names, one per line.
    - Where `train.txt` and `valid.txt` should look as follows, for example.
    ```
    build/darknet/x64/data/img/image1.jpg
    build/darknet/x64/data/img/image2.jpg
    ...
    ```
    - These instructions may also be found in <a href="https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects" target="_blank">How to train on your own data</a>.
2. Upload data to the DSVM as follows.
    - Zip the `data` folder (`zip -r data.zip data` on command line) and copy (`scp data.zip <username>@<public IP or DNS name>:~/darknet/build/darknet/x64/`) the data up to VM (may need to delete networking rule Cleanuptool-Deny-103 again if this gives a timeout error).
    - Log in to the DSVM with SSH
    - In the DSVM, unzip the compressed `data.zip` found, now, in the repo (`darknet`) folder `build/darknet/x64`.
3.  Read through <a href="https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects" target="_blank">How to train on your own data</a>, specifically on updating the `.cfg` files.  We will be using the tiny archicture of YOLO v4.  The following summarizes the changes.
    - Calculate anchor boxes (especially important if you have very big or very small objects on average).  We use `-num_of_clusters 6` because of the tiny architecture which needs 3 anchor box sizes.
        ```
        ./darknet detector calc_anchors build/darknet/x64/data/obj.data -num_of_clusters 6 -width 416 -height 416`
        ```
    - Configure the cfg file (you will see a file called `cfg/yolov4-tiny-custom.cfg`).  Open the file with an editor like `vim` or `nano`.  Modify the following to your scenario.  For example, this header (`net` block):
        ```
        [net]
        # Testing
        #batch=1
        #subdivisions=1
        # Training
        batch=16
        subdivisions=2
        ...

        learning_rate=0.00261
        burn_in=1000
        max_batches = 4000
        policy=steps
        steps=3200,3600
        ...
        ```
        - Info for the `yolo` blocks (in each YOLO block or just before - there are two blocks in the tiny architecture):
            - Class number – change to your number of classes (each YOLO block)
            - Filters – (5 + num_classes)*3  (neural net layer before each YOLO block)
            - Anchors – these are also known as anchor boxes (each YOLO block) - use the calculated anchors from the previous step.
4. Train the model with the following two commands (one downloads the correct pretrained weights for transfer learning (the CNN layers).
    ```
     wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
     
    ./darknet detector train build/darknet/x64/data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -map -dont_show
    ```

## TFlite conversion

- Clone the following repo locally on your local/dev machine.

    `git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git`
- Create a Python 3 environment with `venv` for this project locally (ensure using Python 3).

    ```
    python -m venv env
    ```
- Activate your new Python environment from the command line (unix terminal or Windows `cmd.exe`)

    - Unix systems (Macos, Linux):
    ```
    source env/bin/activate
    ```

    - Windows:
    ```
    env\Scripts\activate.bat
    ```
    - See <a href="https://docs.python.org/3/library/venv.html" target="_blank">venv documenation</a> for more information.
- Install the Python requirements from the `requirements.txt` file in the `tensorflow-yolov4-tflite` folder.
    ```
    pip install -r requirements.txt
    ```

- Download your model to the `tensorflow-yolov4-tflite` by doing following on your local machine with `scp` or similar shell copy tool (on Windows you may use the PuTTy SCP program).

    `scp <username>@<public IP or DNS name>:~/darknet/backup/yolov4-tiny-custom_best.weights .`

- You can use an editor like VSCode or any other text editor will work for the following.
    - Change `coco.names` to `obj.names` in `core/config.py`
    - Place `obj.names` file from your Darknet project in the `data/classes` folder.
- Convert from Darknet to TensorFlow Lite (with quantization) with the two steps as follows.
    ```
    python save_model.py --weights yolov4-tiny-custom_best.weights --output ./checkpoints/yolov4-tiny-416-tflite2 --input_size 416 --model yolov4 --framework tflite --tiny

    python convert_tflite.py --weights ./checkpoints/yolov4-tiny-416-tflite2 --output ./checkpoints/yolov4-tiny-416-fp16.tflite --quantize_mode float16
    ```

- Run the video test locally to check that everything is ok:

    `python detectvideo.py --framework tflite --weights ./checkpoints/yolov4-tiny-416-fp16.tflite --size 416 --tiny --model yolov4 --video 0 --score 0.4`

## Links/references

1. <a href="https://github.com/michhar/darknet-azure-vm" target="_blank">Darknet Azure DSVM</a>
2. <a href="https://github.com/microsoft/VoTT" target="_blank">Visual Object Tagging Tool (VoTT)</a>
3. <a href="https://github.com/AlexeyAB/darknet" target="_blank">Darknet on GitHub</a>
4. <a href="https://docs.python.org/3/library/venv.html" target="_blank">Python virtual environments</a>
5. <a href="https://github.com/hunglc007/tensorflow-yolov4-tflite" target="_blank">Conversion of Darknet model to TFLite on GitHub</a>

## Next steps with LVA

1. <a href="https://github.com/Azure/live-video-analytics/tree/master/utilities/rtspsim-live555" target="_blank">Create a movie simulator docker container with a test video</a>
2. <a href="https://github.com/Azure/live-video-analytics/tree/master/utilities/video-analysis/yolov4-tflite-tiny" target="_blank">TensorFlow Lite Darknet Python AI container sample</a>
3. <a href="https://github.com/Azure-Samples/live-video-analytics-iot-edge-csharp" target="_blank">Run sample app locally</a>
