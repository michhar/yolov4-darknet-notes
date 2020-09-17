# Training a custom YOLO v4 Darknet Model on Azure and Running with Azure Live Video Analytics on IoT Edge

## Train a custom YOLO v4 model

### Prerequisites

- SSH client or command line tool
- Azure Subscription
- Python 3 installed locally
- Familiarity with Unix commands (`vim`, `nano`, etc.)

### Setup on Ubuntu (16.04) DSVM

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

### Train with Darknet

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
3.  Read through <a href="https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects" target="_blank">How to train on your own data</a> from the Darknet repo, mainly on updating the `.cfg` file.  We will be using the tiny archicture of YOLO v4 so will calculate anchors and update the config accordingly (the `cfg/yolov4-tiny-custom.cfg`).  The following summarizes the changes for reference, but please refer to the Darknet repo for more information/clarification.
    - Calculate anchor boxes (especially important if you have very big or very small objects on average).  We use `-num_of_clusters 6` because of the tiny architecture which needs 3 anchor box sizes.  IMPORTANT:  make note of these anchors (darkent creates a file for you called `anchors.txt`) for the section on converting the model to TFLite (you will need them there).
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

### TensorFlow Lite conversion for fast inferencing

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
    - Update the anchors on line 17 of `core/config.py` to match the anchor sizes used to train the model, e.g.:
    ```
    __C.YOLO.ANCHORS_TINY         = [ 81, 27,  28, 80,  58, 51,  76,100, 109, 83,  95,246]
    ```
    - Place `obj.names` file from your Darknet project in the `data/classes` folder.
- Convert from Darknet to TensorFlow Lite (with quantization) with the two steps as follows.
    ```
    python save_model.py --weights yolov4-tiny-custom_best.weights --output ./checkpoints/yolov4-tiny-416-tflite2 --input_size 416 --model yolov4 --framework tflite --tiny

    python convert_tflite.py --weights ./checkpoints/yolov4-tiny-416-tflite2 --output ./checkpoints/yolov4-tiny-416-fp16.tflite --quantize_mode float16
    ```

- Run the video test locally to check that everything is ok:

    `python detectvideo.py --framework tflite --weights ./checkpoints/yolov4-tiny-416-fp16.tflite --size 416 --tiny --model yolov4 --video 0 --score 0.4`

## Set up for Azure Live Video Analytics on IoT Edge

### Prerequisites

On your development machine you will need the following.

1. `git` command line or GUI tool
2. `scp` command line tool (Windows use PuTTy's SCP program)
3. A sample video in `.mkv` format that does not have audio
4. Your `.tflite` model file, anchors and `obj.names` file
    - you can strip audio with FFmpeg (e.g. `ffmpeg -i input_file.mkv -c copy -an output_file.mkv`)
5. Docker
6. VSCode
8. .NET Core 3.1 SDK

On Azure:
- Have gone through the <a href="https://docs.microsoft.com/en-us/azure/media-services/live-video-analytics-edge/get-started-detect-motion-emit-events-quickstart" target="_blank">this Live Video Analytics quickstart</a> to set up the necessary Azure Resources and learn how to use VSCode to see the results
    - OR have the following Azure resources:
        - Azure Container Registry
        - Active Directory Service Principal
        - IoT Hub with an IoT Edge Device
        - Media Services Account
        - Azure IoT Edge VM (Ubuntu Linux)

### Setup for the custom scenario and LVA

1. Create a custom RTSP simulator with your video for inferencing with LVA
    - Clone the official Live Video Analytics GitHub repo:  `git clone https://github.com/Azure/live-video-analytics.git`
    - Open the repository folder in VSCode to make it easier to modify files
    - Go to the RTSP simulator instructions:  `cd utilities/rtspsim-live555/`
    - Replace line 21 with your `.mkv` file (can use ffmpeg to convert from other formats)
        - e.g. `ADD ./your_video_name.mkv /live/mediaServer/media/`
    - Copy your `.mkv` video file to the same folder as Dockerfile
    - Build the docker image according to the Readme
    - Push the docker image to your ACR according to the Readme
2. To prepare the ML model wrapper code, from the base of the live-video-analytics folder:
    - Go to the Docker container building instructions:  `cd utilities/video-analysis/yolov4-tflite-tiny`
    - Copy your `.tflite` model into the `app` folder
    - Perform the following changes to files for your custom scenario:
        - In `app/core/config.py`:
            - Update the `__C.YOLO.ANCHORS_TINY` line to be the same as training Darknet
            - Update the `__C.YOLO.CLASSES` to be `./data/classes/obj.names`
        - In `app/data/classes` folder:
            - Add your file called `obj.names` (with your class names, one per line)
        - In `app/yolov4-tf-tiny-app.py`
            - Update line 31 to use the name of your model
            - Update line 45 to be `obj.names` instead of `coco.names`
        - In the `Dockerfile`
        - We do not need to pull down the yolov4 base tflite model so delete line 19
    - Follow instructions here to build, test, and push to ACR the docker image:
        - https://github.com/Azure/live-video-analytics/tree/master/utilities/video-analysis/yolov4-tflite-tiny
    
## Running the LVA sample app

- To run the sample app and view your inference results:
    - Clone the official Live Video Analytics CSharp sample app: `git clone https://github.com/Azure-Samples/live-video-analytics-iot-edge-csharp.git`
    - Update yolov3.template.json
        - Rename to yolov4.template.json
        - Change the yolov3 name to yolov4
        - Point that yolov4 module to the correct image location in your ACR
        - Point the rtspsim module to the correct image location in your ACR
        - For `rtspsim` module add this to the `createOptions` section:
            ```
            "PortBindings": {
                "554/tcp": [
                    {
                    "HostPort": "5001"
                    }
                ]
            }
            ```
        - Also, in the `rtspsim` module `createOptions` _delete_ the folder bindings, so delete the section like:
            ```
            "HostConfig": {
              "Binds": [
                "$INPUT_VIDEO_FOLDER_ON_DEVICE:/live/mediaServer/media"
              ]
            }
            ```
            - This will ensure that LVA looks in the `rtspsim` module for the video rather than on the IoT Edge device.
    - Make the appropriate changes to the `operations.json`
    - Make the appropriate changes to the `.env` file:
        - Update the `INPUT_VIDEO_FOLDER_ON_DEVICE` to be `/home/<your user name>`
        - Update the `CONTAINER_REGISTRY_USERNAME_myacr` and `CONTAINER_REGISTRY_PASSWORD_myacr`
    - Build the app with `dotnet build`
    - Run the app with `dotnet run`

## Links/references

1. <a href="https://github.com/michhar/darknet-azure-vm" target="_blank">Darknet Azure DSVM</a>
2. <a href="https://github.com/microsoft/VoTT" target="_blank">Visual Object Tagging Tool (VoTT)</a>
3. <a href="https://github.com/AlexeyAB/darknet" target="_blank">Darknet on GitHub</a>
4. <a href="https://docs.python.org/3/library/venv.html" target="_blank">Python virtual environments</a>
5. <a href="https://github.com/hunglc007/tensorflow-yolov4-tflite" target="_blank">Conversion of Darknet model to TFLite on GitHub</a>
6. <a href="https://github.com/Azure/live-video-analytics/tree/master/utilities/rtspsim-live555" target="_blank">Create a movie simulator docker container with a test video for LVA</a>
7. <a href="https://github.com/Azure/live-video-analytics/tree/master/utilities/video-analysis/yolov4-tflite-tiny" target="_blank">TensorFlow Lite Darknet Python AI container sample for LVA</a>
8. <a href="https://github.com/Azure-Samples/live-video-analytics-iot-edge-csharp" target="_blank">Run LVA sample app locally</a>

## Next steps with LVA

