# Cheatsheet for Training a Custom YOLO v4 Darknet Model on Azure and Running with Azure Live Video Analytics on IoT Edge

## Train a custom YOLO v4 model

### Prerequisites

- SSH client or command line tool - for Windows try [putty.exe](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)
- SCP client or command line tool - for Windows try [pscp.exe](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)
- Azure Subscription - a [Free Trial](https://azure.microsoft.com/free/) available for new customers.
- Python 3 installed locally - e.g. [Anaconda Python](https://docs.anaconda.com/anaconda/install/)
- Familiarity with Unix commands - e.g. `vim`, `nano`, `wget`, `curl`, etc.
- Visual Object Tagging Tool - [VoTT](https://github.com/microsoft/VoTT)

### Setup on Ubuntu (16.04) Data Science Virtual Machine and run test

1. Set up an N-series Data Science Virtual Machine (DSVM) with [Darknet](https://github.com/AlexeyAB/darknet) by following <a href="https://github.com/michhar/darknet-azure-vm" target="_blank">these instructions</a>. (IMPORTANT:  ensure authentication with username and password).
2. SSH into the Ubuntu DSVM w/ username and password (of if used ssh key, use that)
    - If this is a corporate subscription, may need to delete an inbound port rule under “Networking” in the Azure Portal (delete Cleanuptool-Deny-103)
3. Test the Darknet executable by running the following.
    - Get the YOLO v4 tiny weights
    ```
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
    ```
    - Run a test on a static image from repository.  Run the following command and then give the path to a test image (look in the `data` folder for sample images e.g. `data/giraffe.jpg`).  The `coco.data` gives the links to other necessary files.  The `yolov4-tiny.cfg` specifies the architecture and settings for tiny YOLO v4.
    ```
    ./darknet detector test ./cfg/coco.data ./cfg/yolov4-tiny.cfg ./yolov4-tiny.weights
    ```
    - Check `predictions.jpg` for the bounding boxes overlaid on the image.  You may "shell copy" (SCP) this file down to your machine to view it or alternatively remote desktop into the machine with a program like [X2Go](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro#x2go).

### Train with Darknet

1. Label some test data locally (aim for about 500-1000 bounding boxes drawn, noting that less will result is less accurate results for those classes)
    - Label data with VoTT and export as `json`
    - Convert the `json` files to YOLO `.txt` files by running the following script (`vott2.0_to_yolo.py`).  In this script, a change must be made.  Update line 13 (`LABELS = {'helmet': 0, 'no_helmet': 1}`) to reflect your classes.  Running this script should result in one `.txt` file per `.json` VoTT annotation file.  The `.txt` files are the YOLO format that `darknet` can use.  Run this conversion script as follows, for example.
    ```
    python vott2.0_to_yolo.py --annot-folder path_to_folder_with_json_files --out-folder new_folder_for_txt_annotations
    ```
    - Darknet will need a specific folder structure.  Structure the data folder as follows where in the `data/img` folder the image is placed along with the `.txt` annotation file.
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
    - `obj.data` is a general file to direct `darknet` to the other data-related files and model folder.  It looks simliar to the following with necessary changes to `classes` for your scenario.
    ```
    classes = 2
    train  = build/darknet/x64/data/train.txt
    valid  = build/darknet/x64/data/valid.txt
    names = build/darknet/x64/data/obj.names
    backup = backup/
    ```
    - `obj.names` contains the class names, one per line.
    - `train.txt` and `valid.txt` should look as follows, for example.  Note, `train.txt` is the training images and is a different subset from the smaller list found in `valid.txt`.  As a general rule, 5-10% of the image paths should be placed in `valid.txt`.  These should be randomly distributed.
    ```
    build/darknet/x64/data/img/image1.jpg
    build/darknet/x64/data/img/image5.jpg
    ...
    ```
    - These instructions may also be found in <a href="https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects" target="_blank">How to train on your own data</a>.
2. Upload data to the DSVM as follows.
    - Zip the `data` folder (`zip -r data.zip data` if using the command line) and copy (`scp data.zip <username>@<public IP or DNS name>:~/darknet/build/darknet/x64/` - use `pscp.exe` on Windows) the data up to VM (may need to delete networking rule Cleanuptool-Deny-103 again if this gives a timeout error).  Note the `data.zip` is placed in the `darknet/build/darknet/x64` folder.  This is where `darknet` will look for the data.
    - Log in to the DSVM with SSH
    - On the DSVM, unzip the compressed `data.zip` found, now, in the folder `darknet/build/darknet/x64`.
3.  Read through <a href="https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects" target="_blank">How to train on your own data</a> from the Darknet repo, mainly on updating the `.cfg` file.  We will be using the tiny archicture of YOLO v4 so will calculate anchors and update the config accordingly (the `cfg/yolov4-tiny-custom.cfg`).  The following summarizes the changes for reference, but please refer to the Darknet repo for more information/clarification.
    - Calculate anchor boxes (especially important if you have very big or very small objects on average).  We use `-num_of_clusters 6` because of the tiny architecture configuration.  IMPORTANT:  make note of these anchors (darknet creates a file for you called `anchors.txt`) for the section on converting the model to TFLite so you will need them later on.
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
4. Train the model with the following two commands.
    ```
     wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
     
    ./darknet detector train build/darknet/x64/data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -map -dont_show
    ```

### TensorFlow Lite conversion for fast inferencing

- Clone the following repo locally on your local/dev machine (this could also be done on the DSVM, but the video detection may be more difficult to view unless remote desktop is being used with DSVM).

    `git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git`
- Create a Python 3 environment with `venv` (virtual environments creation tool) for this project locally (ensure using Python 3).  If `python3` is not available check if `python` points to version 3 and if not, please install Python 3.

    ```
    python3 -m venv env
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

- If you wish to start from this point (do not have a trained model) please refer to the releases (v0.1) for the `.tflite` model, `obj.names` file, anchors (in notes) and sample video (`.mkv` file) to create your RTSP server for simulation:  https://github.com/michhar/yolov4-darknet-notes/releases/tag/v0.1.

### Prerequisites

On your development machine you will need the following.

1. `git` command line tool or client such as [GitHub Desktop](https://desktop.github.com)
2. SCP client or command line tool - for Windows try [pscp.exe](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)
3. A sample video in `.mkv` format (only some audio formats are supported so you may see an error regarding audio format - you may wish to strip audio in this case for the simulator)
4. Your `.tflite` model, anchors and `obj.names` files
5. Docker - such as [Docker Desktop](https://www.docker.com/products/docker-desktop)
6. [VSCode](https://code.visualstudio.com/download) and [Azure IoT Tools extension](https://marketplace.visualstudio.com/items?itemName=vsciot-vscode.azure-iot-tools) (search "Azure IoT Tools" in extensions withing VSCode)
7. .NET Core 3.1 SDK - [download](https://dotnet.microsoft.com/download/dotnet-core/3.1)
8. Azure CLI - [download and install](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
9. `curl` command line tool - [download curl](https://curl.haxx.se/download.html)


On Azure:

- Have gone through the <a href="https://docs.microsoft.com/en-us/azure/media-services/live-video-analytics-edge/get-started-detect-motion-emit-events-quickstart" target="_blank">this Live Video Analytics quickstart</a> and the <a href="https://github.com/Azure-Samples/live-video-analytics-iot-edge-csharp/tree/master/src/cloud-to-device-console-app" target="_blank">Live Video Analytics cloud to device sample console app</a> to set up the necessary Azure Resources and learn how to use VSCode to see the results with .NET app.
    - OR have the following Azure resources provisioned:
        - [Azure Container Registry](https://docs.microsoft.com/en-us/azure/container-registry/)
        - [Active Directory Service Principal](https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal)
        - [IoT Hub](https://docs.microsoft.com/en-us/azure/iot-hub/iot-hub-create-through-portal) with an [IoT Edge Device](https://docs.microsoft.com/en-us/azure/iot-edge/about-iot-edge)
        - [Media Services Account](https://docs.microsoft.com/en-us/azure/media-services/latest/create-account-howto?tabs=portal)
        - [Azure IoT Edge VM (Ubuntu Linux)](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-install-iot-edge-ubuntuvm)

### Create an RTSP simulator

- Create a custom RTSP simulator with your video for inferencing with LVA with live555 media server
    - Clone the official Live Video Analytics GitHub repo:  `git clone https://github.com/Azure/live-video-analytics.git`
    - Open the repository folder in VSCode to make it easier to modify files
    - Go to the RTSP simulator instructions:  `cd utilities/rtspsim-live555/`
    - Replace line 21 with your `.mkv` file (can use the ffmpeg command line tool to convert from other formats like .`mp4` to `.mkv`)
    - Copy your `.mkv` video file to the same folder as Dockerfile
    - Build the docker image according to the Readme
    - Push the docker image to your ACR according to the Readme
        - Login to ACR:  `az acr login --name myregistry`
        - Use docker to push: `docker push myregistry.azurecr.io/my-rtsp-sim:latest`

### Create the AI container for inferencing

- To prepare the ML model wrapper code, from the base of the live-video-analytics folder:
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
    - In the `src/edge` folder, update `yolov3.template.json` as follows.
        - Rename to `yolov4.template.json`
        - Update (or ensure this is the case) the `runtime` at the beginning of the file looks like:
            ```
            "runtime": {
                "type": "docker",
                "settings": {
                "minDockerVersion": "v1.25",
                "loggingOptions": "",
                    "registryCredentials": {
                          "$CONTAINER_REGISTRY_USERNAME_myacr": {
                                "username": "$CONTAINER_REGISTRY_USERNAME_myacr",
                                "password": "$CONTAINER_REGISTRY_PASSWORD_myacr",
                                "address": "$CONTAINER_REGISTRY_USERNAME_myacr.azurecr.io"
                          }
                    }
                }
            }
            ```
            - This section will ensure the deployment can find your custom `rtspsim` and `yolov4` images in your ACR.
        - Change the `yolov3` name to `yolov4` as in the following modules section (the image location is an example) pointing the yolov4 module to the correct image location in your ACR.
            ```
            "yolov4": {
                "version": "1.0",
                "type": "docker",
                "status": "running",
                "restartPolicy": "always",
                "settings": {
                  "image": "myacr.azurecr.io/my-awesome-custom-yolov4:latest",
                      "createOptions": {}
                }
          }
            ```
        - For `rtspsim` module ensure the image points to your image in ACR (the image location is an example) and ensure the `createOptions` look as follows:
            ```
              "rtspsim": {
                "version": "1.0",
                "type": "docker",
                "status": "running",
                "restartPolicy": "always",
                "settings": {
                  "image": "myacr.azurecr.io/my-rtsp-sim:latest",
                  "createOptions": {

                    "PortBindings": {
                        "554/tcp": [
                            {
                                "HostPort": "5001"
                            }
                        ]
                     }
                   }
                 }
               }
            ```
        - Also, in the `rtspsim` module `createOptions` make sure to _delete_ the folder bindings, so delete any section like:
            ```
            "HostConfig": {
              "Binds": [
                "$INPUT_VIDEO_FOLDER_ON_DEVICE:/live/mediaServer/media"
              ]
            }
            ```
            - This will ensure that LVA looks in the `rtspsim` module for the video rather than on the IoT Edge device.
    - Make the appropriate changes to the `.env` file (this should be located in the `src/edge` folder:
        - Update the `CONTAINER_REGISTRY_USERNAME_myacr` and `CONTAINER_REGISTRY_PASSWORD_myacr`
        - Recall the `.env` file (you can modify in VSCode) should have the following format (fill in the missing parts for your Azure resources):
            ```
            SUBSCRIPTION_ID=
            RESOURCE_GROUP=
            AMS_ACCOUNT=
            IOTHUB_CONNECTION_STRING=
            AAD_TENANT_ID=
            AAD_SERVICE_PRINCIPAL_ID=
            AAD_SERVICE_PRINCIPAL_SECRET=
            INPUT_VIDEO_FOLDER_ON_DEVICE="/live/mediaServer/media"
            OUTPUT_VIDEO_FOLDER_ON_DEVICE="/var/media"
            APPDATA_FOLDER_ON_DEVICE="/var/lib/azuremediaservices"
            CONTAINER_REGISTRY_USERNAME_myacr=
            CONTAINER_REGISTRY_PASSWORD_myacr=
            ```
            - When you create the manifest template file in VSCode it will use these values to create the actual deployment manifest file.
    - In the `src/cloud-to-device-console-app` folder, make the appropriate changes to the `operations.json`.
        - In the `"opName": "GraphTopologySet"`, update the `topologyUrl` to be the http extension topology as follows.
        ```
        {
            "opName": "GraphTopologySet",
            "opParams": {
                "topologyUrl": "https://raw.githubusercontent.com/Azure/live-video-analytics/master/MediaGraph/topologies/httpExtension/topology.json"
            }
        }
        ```
        - In the `"opName": "GraphInstanceSet"`, update the `rtspUrl` value to have your video file name (here `my_video.mkv`) and `inferencingUrl` with `"value": "http://yolov4/score"`, as in:
        ```
        {
            "opName": "GraphInstanceSet",
            "opParams": {
                "name": "Sample-Graph-1",
                "properties": {
                    "topologyName" : "InferencingWithHttpExtension",
                    "description": "Sample graph description",
                    "parameters": [
                        {
                            "name": "rtspUrl",
                            "value": "rtsp://rtspsim:554/media/my_video.mkv"
                        },
                        {
                            "name": "rtspUserName",
                            "value": "testuser"
                        },
                        {
                            "name": "rtspPassword",
                            "value": "testpassword"
                        },
                        {
                            "name": "imageEncoding",
                            "value": "jpeg"
                        },
                        {
                            "name": "inferencingUrl",
                            "value": "http://yolov4/score"
                        }
                    ]
                }
            }
        },
        ```
    - Make the appropriate changes to the `appsettings.json`, a file that you may need to create if you haven't done the quickstarts.  It should look as follows and be located in the `src/cloud-to-device-console-app` folder.
        ```
        {
            "IoThubConnectionString" : "connection_string_of_iothub",
            "deviceId" : "name_of_your_edge_device_in_iot_hub",
            "moduleId" : "lvaEdge"
        }
        ```
        - The IoT Hub connection string may be found in the Azure Portal under your IoT Hub -> Settings -> Shared access policies blade -> iothubowner Policy -> Connection string—primary key
    - Build the app with `dotnet build` from the `src/cloud-to-device-console-app` folder.
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

