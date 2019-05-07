#!/usr/bin/env python3.6

import argparse
import os

import affvisionpy as af
import cv2 as cv2

# Constants
MAX_NUM_OF_FACES = 10
NOT_A_NUMBER = 'NaN'
count = 0
measurements_dict = dict()


class Listener(af.ImageListener):
    def __init__(self):
        super(Listener, self).__init__()

    def results_updated(self, faces, image):
        self.faces = faces
        for fid, face in faces.items():
            measurements_dict = face.get_measurements()
            # print(measurements_dict)
            expresions_dict = face.get_expressions()
            # print(expresions_dict)
            emotions_dict = face.get_emotions()
            # print(emotions_dict)

    def image_captured(self, image):
        pass


def get_command_line_parameters(args):
    input_file = args.input
    if not os.path.isfile(input_file):
        raise ValueError("Please provide a valid input file")
    data = args.data
    if not os.path.isdir(data):
        raise ValueError("Please check your data file path")

    return input_file, data


def run():
    args = parse_command_line()
    input_file, data = get_command_line_parameters(args)
    detector = af.FrameDetector(data, MAX_NUM_OF_FACES, 10)

    detector.enable_feature({af.Feature.expressions, af.Feature.emotions})

    list = Listener()
    detector.set_image_listener(list)

    detector.start()

    if not os.path.isdir("opvideo"):
        os.mkdir("opvideo")

    captureFile = cv2.VideoCapture(input_file)
    count = 0

    while captureFile.isOpened():
        # Capture frame-by-frame
        ret, frame = captureFile.read()

        if ret == True:
            print('Read %d frame: ' % count, ret)
            height = frame.shape[0]
            width = frame.shape[1]
            timestamp = int(captureFile.get(cv2.CAP_PROP_POS_MSEC))

            afframe = af.Frame(width, height, frame, af.ColorFormat.bgr, timestamp)
            count += 1

            try:
                detector.process(afframe)
                cv2.imwrite(os.path.join("opvideo", "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file

            except Exception as exp:
                print(exp)
        else:
            break

    captureFile.release()
    detector.stop()


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", dest="data", required=True, help="path to directory containing the models")
    parser.add_argument("-i", "--input", dest="input", required=True, help="path to input video file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    run()
