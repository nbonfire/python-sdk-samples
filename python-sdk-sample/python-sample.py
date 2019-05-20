#!/usr/bin/python
# !/usr/bin/env python3.6
import argparse
import csv
import os
from collections import defaultdict

import affvisionpy as af
import cv2 as cv2
import math

# Constants
NOT_A_NUMBER = 'NaN'
count = 0
TEXT_SIZE = 0.4
PADDING_FOR_SEPARATOR = 5
THRESHOLD_VALUE_FOR_EMOTIONS = 5

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
            global mood
            mood = str(face.get_mood())
            global dominant_emotion
            dominant_emotion = str(face.get_dominant_emotion().dominant_emotion)
            print(dominant_emotion)
            bounding_box_dict[face.get_id()] = [face.get_bounding_box()[0].x,
                                                face.get_bounding_box()[0].y,
                                                face.get_bounding_box()[1].x,
                                                face.get_bounding_box()[1].y]

    def image_captured(self, image):
        pass


def get_command_line_parameters(args):
    input_file = args.input

    if input_file == "camera":
        input_file = 0
    else:
        if not os.path.isfile(input_file):
            raise ValueError("Please provide a valid input file")
    data = args.data
    if not os.path.isdir(data):
        raise ValueError("Please check your data file path")
    max_num_of_faces = int(args.num_faces)

    return input_file, data, max_num_of_faces


def draw_bounding_box(frame):
    for fid, bb_points in bounding_box_dict.items():
        x1, y1, x2, y2 = get_bounding_box_points(fid)
        for key in emotions_dict[fid]:
            if 'valence' in str(key):
                valence_value = emotions_dict[fid][key]
            if 'anger' in str(key):
                anger_value = emotions_dict[fid][key]
            if 'joy' in str(key):
                joy_value = emotions_dict[fid][key]
        if valence_value < 0 and anger_value >= THRESHOLD_VALUE_FOR_EMOTIONS:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        elif valence_value >= THRESHOLD_VALUE_FOR_EMOTIONS and joy_value >= THRESHOLD_VALUE_FOR_EMOTIONS:
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


def get_text_size(text, font, thickness):
    text_size = cv2.getTextSize(text, font, TEXT_SIZE, thickness)
    return text_size[0][0], text_size[0][1]


def display_measurements_on_screen(key, val, upper_left_y, frame, x1):
    key = str(key)

    key_name = key.split(".")[1]
    key_text_width, key_text_height = get_text_size(key_name, cv2.FONT_HERSHEY_SIMPLEX, 1)
    val_text = str(round(val, 2))
    val_text_width, val_text_height = get_text_size(val_text, cv2.FONT_HERSHEY_SIMPLEX, 1)

    key_val_width = key_text_width + val_text_width

    cv2.putText(frame, key_name + ": ", (abs(x1 - key_val_width), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SIZE,
                (255, 255, 255))
    cv2.putText(frame, val_text, (abs(x1 - val_text_width + PADDING_FOR_SEPARATOR), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,
                (255, 255, 255))


def display_mood_and_dominant_emotion_on_screen(attribute, upper_left_y, frame, x1):
    attribute_key_name = attribute.split(".")[0]
    attribute_text_width, attribute_text_height = get_text_size(attribute_key_name, cv2.FONT_HERSHEY_SIMPLEX, 1)
    attribute_val = attribute.split(".")[1]
    val_text_width, val_text_height = get_text_size(attribute_val, cv2.FONT_HERSHEY_SIMPLEX, 1)

    key_val_width = attribute_text_width + val_text_width

    cv2.putText(frame, attribute_key_name + ": ", (abs(x1 - key_val_width), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SIZE,
                (255, 255, 255))
    cv2.putText(frame, attribute_val, (abs(x1 - val_text_width + PADDING_FOR_SEPARATOR), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,
                (255, 255, 255))


def display_emotions_on_screen(key, val, upper_left_y, frame, x1):
    key = str(key)
    key_name = key.split(".")[1]
    key_text_width, key_text_height = get_text_size(key_name, cv2.FONT_HERSHEY_SIMPLEX, 1)

    val_rect_width = 120
    key_val_width = key_text_width + val_rect_width
    cv2.putText(frame, key_name + ": ", (abs(x1 - key_val_width), upper_left_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SIZE,
                (255, 255, 255))
    overlay = frame.copy()

    if math.isnan(val):
        val = 0

    start_box_point_x = abs(x1 - val_rect_width)
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


def display_expressions_on_screen(key, val, upper_right_x, upper_right_y, frame, upper_left_y):
    key = str(key)

    key_name = key.split(".")[1]
    val_rect_width = 120
    overlay = frame.copy()
    if math.isnan(val):
        val = 0

    if 'blink' not in key:
        start_box_point_x = upper_right_x
        width = 8
        height = 10

        rounded_val = roundup(val)
        rounded_val /= 10
        rounded_val = int(rounded_val)
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
        cv2.putText(frame, str(val), (upper_right_x, upper_right_y), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,
                    (255, 255, 255))

    cv2.putText(frame, " :" + str(key_name), (upper_right_x + val_rect_width, upper_right_y), cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SIZE,
                (255, 255, 255))


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
            display_measurements_on_screen(key, val, upper_left_y, frame, x1)

            upper_left_y += 25

        for key, val in emotions.items():
            display_emotions_on_screen(key, val, upper_left_y, frame, x1)
            upper_left_y += 25

        for key, val in expressions.items():
            display_expressions_on_screen(key, val, upper_right_x, upper_right_y, frame, upper_left_y)

            upper_right_y += 25

        display_mood_and_dominant_emotion_on_screen(mood, upper_left_y, frame, x1)
        upper_left_y += 25
        display_mood_and_dominant_emotion_on_screen(dominant_emotion, upper_left_y, frame, x1)


def run(csv_data):
    args = parse_command_line()
    input_file, data, max_num_of_faces = get_command_line_parameters(args)
    detector = af.SyncFrameDetector(data, max_num_of_faces)

    detector.enable_features({af.Feature.expressions, af.Feature.emotions})

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

            if len(num_faces) > 0 and check_bounding_box_outside(width, height) == False:
                draw_bounding_box(frame)
                draw_affectiva_logo(frame, width, height)
                write_metrics(frame)
                out.write(frame)
                cv2.imshow('Frame', frame)
                cv2.imwrite(os.path.join("opvideo", "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
            else:
                clear_all_dictionaries()
                draw_affectiva_logo(frame, width, height)
                cv2.imshow('Frame', frame)
                cv2.imwrite(os.path.join("opvideo", "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file

            if cv2.waitKey(1) == 27:
                break
        else:
            break

    captureFile.release()
    cv2.destroyAllWindows()
    detector.stop()
    write_csv_data_to_file(csv_data)


def clear_all_dictionaries():
    bounding_box_dict.clear()
    emotions_dict.clear()
    expressions_dict.clear()
    measurements_dict.clear()


def draw_affectiva_logo(frame, width, height):
    logo = cv2.imread("Final logo - RGB Magenta.png")
    logo_width = int(width / 3)
    logo_height = int(height / 10)
    logo = cv2.resize(logo, (logo_width, logo_height))

    y1, y2 = 0, logo_height
    x1, x2 = width - logo_width, width
    # Remove the white background from the logo so that only the word "Affectiva" is visible on screen
    for c in range(0, 3):
        alpha = logo[0:logo_height, 0:logo_width, 1] / 255.0
        color = logo[0:logo_height, 0:logo_width, c] * (1.0 - alpha)
        beta = frame[y1:y2, x1:x2, c] * (alpha)
        frame[y1:y2, x1:x2, c] = color + beta


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
    parser.add_argument("-i", "--input", dest="input", required=False, default="camera",
                        help="path to input video file")
    parser.add_argument("-n", "--num_faces", dest="num_faces", required=False, default=1,
                        help="number of faces to identify in the frame")
    args = parser.parse_args()
    return args


def write_csv_data_to_file(csv_data):
    header_row = ['TimeStamp', 'faceId', 'upperLeftX', 'upperLeftY', 'lowerRightX', 'lowerRightY', 'Pitch', 'Yaw',
                  'Roll', 'interocularDistance', 'joy', 'anger', 'surprise',
                  'valence',
                  'fear', 'disgust', 'sadness', 'neutral', 'smile', 'browRaise', 'browFurrow', 'noseWrinkle',
                  'upperLipRaise',
                  'mouthOpen', 'eyeClosure', 'cheekRaise', 'eyeWiden', 'innerBrowRaise', 'lipCornerDepressor',
                  'yawn', 'blink', 'blinkRate']

    with open('output.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        writer.writerows([header_row])
        for row in csv_data:
            writer.writerows([row])

        csv_file.close()


if __name__ == "__main__":
    csv_data = list()
    run(csv_data)

