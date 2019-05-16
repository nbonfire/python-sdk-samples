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
        global num_faces
        num_faces = faces
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


def draw_bounding_box(frame, width, height):
    for fid, bb_points in bounding_box_dict.items():
        x1, y1, x2, y2 = get_bounding_box_points(fid)
        for key in emotions_dict[fid]:
            if 'valence' in str(key):
                valence_value = emotions_dict[fid][key]
            if 'anger' in str(key):
                anger_value = emotions_dict[fid][key]
            if 'joy' in str(key):
                joy_value = emotions_dict[fid][key]
        if valence_value < 0 and anger_value >= 5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        elif valence_value >= 5 and joy_value >= 5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (21, 169, 167), 3)


def get_bounding_box_points(fid):
    return (int(bounding_box_dict[fid][0]),
            int(bounding_box_dict[fid][1]),
            int(bounding_box_dict[fid][2]),
            int(bounding_box_dict[fid][3]))


def roundup(num):
    return int(math.ceil(num / 10.0)) * 10


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
        left_padding = 20
        for key, val in measurements.items():
            key = str(key)
            cv2.putText(frame, str(key.split(".")[1]) + ":", (abs(upper_left_x - left_padding), upper_left_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255))
            cv2.putText(frame, str(round(val, 2)), (abs(upper_left_x - left_padding) + 130, upper_left_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255))

            upper_left_y += 25
        for key, val in emotions.items():
            key = str(key)
            cv2.putText(frame, str(key.split(".")[1]) + ":", (abs(upper_left_x - left_padding), upper_left_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255))
            overlay = frame.copy()

            if math.isnan(val):
                val = 0

            start_box_point_x = abs(upper_left_x - left_padding) + 50
            width = 8
            height = 10

            rounded_val = roundup(val)
            rounded_val /= 10
            rounded_val = abs(int(rounded_val))

            for i in range(0, rounded_val + 1):
                start_box_point_x += 10
                cv2.rectangle(overlay, (start_box_point_x, upper_left_y),
                              (start_box_point_x + width, upper_left_y - height), (186, 186, 186), -1)
                if ('valence' in key and val < 0) or ('anger' in key and val > 0):
                    cv2.rectangle(overlay, (start_box_point_x, upper_left_y),
                                  (start_box_point_x + width, upper_left_y - height), (0, 0, 255), -1)
                else:
                    cv2.rectangle(overlay, (start_box_point_x, upper_left_y),
                                  (start_box_point_x + width, upper_left_y - height), (0, 204, 102), -1)
            for i in range(rounded_val + 1, 11):
                start_box_point_x += 10
                cv2.rectangle(overlay, (start_box_point_x, upper_left_y),
                              (start_box_point_x + width, upper_left_y - height), (186, 186, 186), -1)

            alpha = 0.8
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            upper_left_y += 25
        for key, val in expressions.items():
            key = str(key)
            cv2.putText(frame, str(key.split(".")[1]) + ":", (upper_right_x, upper_right_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255))

            overlay = frame.copy()
            if math.isnan(val):
                val = 0

            if 'blink' not in key:
                start_box_point_x = upper_right_x + 100
                width = 8
                height = 10

                rounded_val = roundup(val)
                rounded_val /= 10
                rounded_val = int(rounded_val)
                # print(rounded_val)
                for i in range(0, rounded_val + 1):
                    start_box_point_x += 10
                    cv2.rectangle(overlay, (start_box_point_x, upper_right_y),
                                  (start_box_point_x + width, upper_right_y - height), (186, 186, 186), -1)
                    cv2.rectangle(overlay, (start_box_point_x, upper_right_y),
                                  (start_box_point_x + width, upper_right_y - height), (0, 204, 102), -1)
                for i in range(rounded_val + 1, 11):
                    start_box_point_x += 10
                    cv2.rectangle(overlay, (start_box_point_x, upper_right_y),
                                  (start_box_point_x + width, upper_right_y - height), (186, 186, 186), -1)

                alpha = 0.8
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                upper_left_y += 25
            else:
                cv2.putText(frame, str(val), (upper_right_x + 100, upper_right_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255))
            upper_right_y += 25


def run(csv_data):
    args = parse_command_line()
    input_file, data = get_command_line_parameters(args)
    detector = af.SyncFrameDetector(data, MAX_NUM_OF_FACES)

    detector.enable_features({af.Feature.expressions, af.Feature.emotions})

    list = Listener()
    detector.set_image_listener(list)

    detector.start()

    if not os.path.isdir("opvideo"):
        os.mkdir("opvideo")

    captureFile = cv2.VideoCapture(0)

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
            print(height)
            print(width)
            timestamp = int(captureFile.get(cv2.CAP_PROP_POS_MSEC))

            afframe = af.Frame(width, height, frame, af.ColorFormat.bgr, timestamp)
            count += 1
            try:
                detector.process(afframe)
            except Exception as exp:
                print(exp)

            write_metrics_to_csv_data_list(csv_data, timestamp)

            if len(num_faces) > 0 and check_bounding_box_outside(width,height) == False:
                draw_bounding_box(frame, width, height)

                write_metrics(frame)
                out.write(frame)
                cv2.imshow('Frame', frame)
                cv2.imwrite(os.path.join("opvideo", "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
            else:
                bounding_box_dict.clear()
                emotions_dict.clear()
                expressions_dict.clear()
                measurements_dict.clear()
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


def check_bounding_box_outside(width, height):
    for fid in bounding_box_dict.keys():
        x1, y1, x2, y2 = get_bounding_box_points(fid)
        if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
            return True
        return False


def write_metrics_to_csv_data_list(csv_data, timestamp):
    for fid in measurements_dict.keys():
        current_frame_data = list()
        current_frame_data.append(timestamp)
        current_frame_data.append(fid)
        for val in bounding_box_dict[fid]:
            current_frame_data.append(round(val, 4))
        for val in measurements_dict[fid].values():
            current_frame_data.append(round(val, 4))
        for val in emotions_dict[fid].values():
            current_frame_data.append(round(val, 4))
        for val in expressions_dict[fid].values():
            current_frame_data.append(round(val, 4))
        csv_data.append(current_frame_data)


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", dest="data", required=True, help="path to directory containing the models")
    parser.add_argument("-i", "--input", dest="input", required=True, help="path to input video file")
    args = parser.parse_args()
    return args


def write_csv_data_to_file(csv_data, input_file):
    header_row = ['TimeStamp', 'faceId', 'upperLeftX', 'upperLeftY', 'lowerRightX', 'lowerRightY', 'Pitch', 'Yaw',
                  'Roll', 'interocularDistance', 'joy', 'anger', 'surprise',
                  'valence',
                  'fear', 'disgust', 'sadness', 'neutral', 'smile', 'browRaise', 'browFurrow', 'noseWrinkle',
                  'upperLipRaise',
                  'mouthOpen', 'eyeClosure', 'cheekRaise', 'eyeWiden', 'innerBrowRaise', 'lipCornerDepressor',
                  'yawn', 'blink', 'blinkRate']

    with open('output.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        # writer.writerows([["Input file: " + input_file.rsplit(os.sep, 1)[1]]])
        writer.writerows([header_row])
        for row in csv_data:
            writer.writerows([row])

        csv_file.close()


if __name__ == "__main__":
    csv_data = list()
    run(csv_data)
