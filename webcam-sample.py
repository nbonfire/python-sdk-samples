#!/usr/bin/env python3.6

import argparse
import csv
import os
from collections import defaultdict

import affvisionpy as af
import cv2 as cv2
import math

# Constants
MAX_NUM_OF_FACES = 10
NOT_A_NUMBER = 'NaN'
count = 0

measurements_dict = defaultdict()
expressions_dict = defaultdict()
emotions_dict = defaultdict()
bounding_box_dict = defaultdict()


class Listener(af.ImageListener):
    def __init__(self):
        super(Listener, self).__init__()

    def results_updated(self, faces, image):
        self.faces = faces
        for fid, face in faces.items():
            measurements_dict[face.get_id()] = defaultdict()
            expressions_dict[face.get_id()] = defaultdict()
            emotions_dict[face.get_id()] = defaultdict()
            measurements_dict[face.get_id()].update(face.get_measurements())
            expressions_dict[face.get_id()].update(face.get_expressions())
            emotions_dict[face.get_id()].update(face.get_emotions())
            bounding_box_dict[face.get_id()] = [face.get_bounding_box()[0].x,
                                                face.get_bounding_box()[0].y,
                                                face.get_bounding_box()[1].x,
                                                face.get_bounding_box()[1].y]

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


def draw_bounding_box(frame):
    for fid, bb_points in bounding_box_dict.items():
        x1, y1, x2, y2 = get_bounding_box_points(fid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)


def get_bounding_box_points(fid):
    return (int(bounding_box_dict[fid][0]),
            int(bounding_box_dict[fid][1]),
            int(bounding_box_dict[fid][2]),
            int(bounding_box_dict[fid][3]))


def write_metrics(frame):
    for fid in measurements_dict.keys():
        measurements = measurements_dict[fid]
        expressions = expressions_dict[fid]
        emotions = emotions_dict[fid]
        x1, y1, x2, y2 = get_bounding_box_points(fid)
        box_height = y2 - y1
        box_width = x2 - x1
        upper_left_x = abs(box_width - x1)
        upper_left_y = y1
        upper_right_x = x1 + box_width
        upper_right_y = abs(y2 - box_height)
        for key, val in measurements.items():
            key = str(key)
            cv2.putText(frame, str(key.split(".")[1]) + ":", (upper_left_x, upper_left_y), cv2.FONT_HERSHEY_TRIPLEX,
                        0.3,
                        (255, 255, 255))
            cv2.putText(frame, str(val), (upper_left_x + 160, upper_left_y), cv2.FONT_HERSHEY_TRIPLEX, 0.3,
                        (255, 255, 255))
            # if math.isnan(val):
            #     val = 0
            # cv2.rectangle(frame, (upper_left_x + 160, upper_left_y), (upper_left_x + 160 + int(val), upper_left_y -
            # 10),
            #               (255, 255, 255),
            #               -1)

            upper_left_y += 25
        for key, val in emotions.items():
            key = str(key)
            cv2.putText(frame, str(key.split(".")[1]) + ":", (upper_left_x, upper_left_y), cv2.FONT_HERSHEY_TRIPLEX,
                        0.3,
                        (255, 255, 255))
            # cv2.putText(frame, str(val), (upper_left_x + 160, upper_left_y), cv2.FONT_HERSHEY_TRIPLEX, 0.3,
            #             (255, 255, 255))
            overlay = frame.copy()
            if math.isnan(val):
                val = 0
            cv2.rectangle(overlay, (upper_left_x + 160, upper_left_y),
                          (upper_left_x + 160 + int(val), upper_left_y - 10),
                          (255, 255, 255),
                          -1)
            alpha = 0.8
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            upper_left_y += 25
        for key, val in expressions.items():
            key = str(key)
            cv2.putText(frame, str(key.split(".")[1]) + ":", (upper_right_x, upper_right_y), cv2.FONT_HERSHEY_TRIPLEX,
                        0.3,
                        (255, 255, 255))

            overlay = frame.copy()
            if math.isnan(val):
                val = 0

            if 'blink' not in key:
                cv2.rectangle(overlay, (upper_right_x + 160, upper_right_y),
                              (upper_right_x + 160 + int(val * 100), upper_right_y - 10),
                              (255, 255, 255),
                              -1)
                alpha = 0.8
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            else:
                cv2.putText(frame, str(val), (upper_right_x + 160, upper_right_y), cv2.FONT_HERSHEY_TRIPLEX, 0.3,
                            (255, 255, 255))
            upper_right_y += 25


def run(csv_data):
    args = parse_command_line()
    input_file, data = get_command_line_parameters(args)
    detector = af.FrameDetector(data, MAX_NUM_OF_FACES)

    detector.enable_feature({af.Feature.expressions, af.Feature.emotions})

    list = Listener()
    detector.set_image_listener(list)

    detector.start()

    if not os.path.isdir("opvideo"):
        os.mkdir("opvideo")

    captureFile = cv2.VideoCapture(input_file)

    file_width = int(captureFile.get(3))
    file_height = int(captureFile.get(4))
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (file_width, file_height))
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
            except Exception as exp:
                print(exp)

            write_metrics_to_csv_data_list(csv_data, timestamp)

            draw_bounding_box(frame)

            write_metrics(frame)
            out.write(frame)
            cv2.imshow('Frame', frame)
            cv2.imwrite(os.path.join("opvideo", "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file

            if cv2.waitKey(1) == 27:
                break
        else:
            break

    captureFile.release()
    cv2.destroyAllWindows()
    detector.stop()
    write_csv_data_to_file(csv_data, input_file)


def write_metrics_to_csv_data_list(csv_data, timestamp):
    for fid in measurements_dict.keys():
        current_frame_data = list()
        current_frame_data.append(timestamp)
        current_frame_data.append(fid)
        for val in measurements_dict[fid].values():
            current_frame_data.append(val)
        for val in emotions_dict[fid].values():
            current_frame_data.append(val)
        for val in expressions_dict[fid].values():
            current_frame_data.append(val)
        csv_data.append(current_frame_data)


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", dest="data", required=True, help="path to directory containing the models")
    parser.add_argument("-i", "--input", dest="input", required=True, help="path to input video file")
    args = parser.parse_args()
    return args


def write_csv_data_to_file(csv_data, input_file):
    header_row = ['Timestamp', 'Face ID', 'Pitch', 'Yaw', 'Roll', 'Interocular distance', 'joy', 'anger', 'surprise',
                  'valence',
                  'fear', 'disgust', 'sadness', 'neutral', 'smile', 'brow raise', 'brow furrow', 'nose wrinkle',
                  'upper lip raise',
                  'mouth open', 'eye closure', 'cheek raise', 'eye widen', 'inner brow raise', 'lip corner depressor',
                  'yawn', 'blink', 'blink_rate']

    with open('output.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        writer.writerows([["Input file: " + input_file.rsplit(os.sep, 1)[1]]])
        writer.writerows([header_row])
        for row in csv_data:
            writer.writerows([row])

        csv_file.close()


if __name__ == "__main__":
    csv_data = list()
    run(csv_data)
