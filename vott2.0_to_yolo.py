"""Convert the annotations from using VoTT 2.0 labeling tool
 to YOLO text format for this project.
 
 Also, creates a test.txt and train.txt file with paths
 to test and train images."""
import argparse
import json
import glob
import os
import random

# Change to match your labels and give a unique number starting at 0
LABELS = {'helmet': 0, 'no_helmet': 1}

def convert2yolo(annotarray, origwidth, origheight):
    """
    Convert VoTT json format to yolo
    
    height,width,left,top --> x_center,y_center,width,height
    * NOTE:  the "top" here is distance from the top.
    Arguments
    ---------
    annotarray :  list
        [height,width,left,top]
    Returns
    -------
    list
        bounding box in format [x_center,y_center,width,height]
    """
    boxheight = float(annotarray[0])
    boxwidth = float(annotarray[1])
    boxleft = float(annotarray[2])
    boxtop = float(annotarray[3])

    # Calculate x center and y center, then scale 0-1
    x_center = (boxleft + (0.5*boxwidth))/origwidth
    y_center = (boxtop + (0.5*boxheight))/origheight

    # Scale box width, height to 0-1
    boxwidth, boxheight = boxwidth/origwidth, boxheight/origheight

    return [x_center, y_center, boxwidth, boxheight]

def getannot(annotdict):
    """
    Convert the json annot for one image to proper yolo format
    * NOTE:  the "top" here is distance from the top.
    Arguments
    ---------
    annotdict : dict
        json dictionary with data for one image
    """
    filename = annotdict['asset']['name']
    origwidth = annotdict['asset']['size']['width']
    origheight = annotdict['asset']['size']['height']

    regions = annotdict['regions']
    regions2 = []
    for r in regions:
        label = LABELS[r['tags'][0]] #Should only be one
        bbox = r['boundingBox']
        h = bbox['height']
        w = bbox['width']
        left = bbox['left']
        top = bbox['top'] # distance from top
        yoloarray = [label]
        yoloarray.extend(convert2yolo([h, w, left, top], origwidth, origheight))
        regions2.append(yoloarray)
    return regions2, filename

def extractannots(filelist, outdir):
    """Operates over all json files to extract annotations
    Writes the yolo format .txt files to specified location"""

    justfilenames = []

    for file in filelist:
        with open(file, 'r') as fptr:
            filedata = json.load(fptr)
            yoloregions, filename = getannot(filedata)
        ending = filename.split('.')[-1]
        with open(os.path.join(outdir, filename.replace(ending, 'txt')), 'w') as fptr:
            for annot in yoloregions:
                annot = [str(a) for a in annot]
                fptr.write(' '.join(annot) + '\n')
        justfilenames.append(filename)

        trainset = []
        testset = []
        for f in justfilenames:
            r = random.choice(range(10))
            if r < 2:
                testset.append(os.path.join('data', 'obj', f))
            else:
                trainset.append(os.path.join('data', 'obj', f))
        with open('train.txt', 'w') as fptr:
            for samp in trainset:
                fptr.write(samp + '\n')
        with open('test.txt', 'w') as fptr:
            for samp in testset:
                fptr.write(samp + '\n')

if __name__ == "__main__":
    """Main"""
    # For command line options
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    # There should be one "asset.json" for each image with annotations
    parser.add_argument(
        '--annot-folder', type=str, dest='annot_folder', default='vott',
        help='Annotations folder from VoTT 2.0 json format export'
    )
    # There should be one "asset.json" for each image with annotations
    parser.add_argument(
        '--out-folder', type=str, dest='out_folder', default='convert_out',
        help='Output folder - will overwrite!'
    )

    args = parser.parse_args()

    json_annot_files = glob.glob(os.path.join(args.annot_folder, '*-asset.json'))
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    extractannots(json_annot_files, args.out_folder)
