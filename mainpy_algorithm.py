import json
import numpy as np
import cv2
import onnxruntime
import math
import copy

model = "./tinyyolov2-8.onnx"
path = "./2.JPG"

# Preprocess the image
img = cv2.imread(path)
img = cv2.resize(img, dsize=(416, 416))
img = img.swapaxes(0, 2)
img = img.swapaxes(1, 2)
img = np.expand_dims(img, axis=0)

data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

result = session.run([output_name], {input_name: data})

result_list = result[0].reshape([21125]).tolist()

YOLO_NUM_CLASSES = 20
YOLO_ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
YOLO_SIDE = 13
YOLO_NUM_BOXES = 5

THRESHOLD = 0.3


def compute_prediction(model_result, threshold=THRESHOLD):
    """
    post process of tiny yolo v2 output
    :param model_result: raw model output list [21125]
    :param threshold: prediction threshold
    :return: detected objects list
    """
    predictions = []

    for cy in range(YOLO_SIDE):
        for cx in range(YOLO_SIDE):
            for b in range(YOLO_NUM_BOXES):
                channel = b * (YOLO_NUM_CLASSES + 5)
                tx = model_result[compute_entry(channel, cy, cx)]
                ty = model_result[compute_entry(channel + 1, cy, cx)]
                tw = model_result[compute_entry(channel + 2, cy, cx)]
                th = model_result[compute_entry(channel + 3, cy, cx)]
                tc = model_result[compute_entry(channel + 4, cy, cx)]

                # apply sigmoid function to raw data
                x = (cx + sigmoid(tx)) * 32
                y = (cy + sigmoid(ty)) * 32
                w = math.exp(tw) * YOLO_ANCHORS[2 * b] * 32
                h = math.exp(th) * YOLO_ANCHORS[2 * b + 1] * 32
                confidence = sigmoid(tc)

                class_scores = [model_result[compute_entry(channel + 5 + idx, cx, cy)] for idx in range(YOLO_NUM_CLASSES)]
                class_scores = softmax(class_scores)

                best_class_score = max(class_scores)
                detected_class = class_scores.index(best_class_score)
                class_confidence = best_class_score * confidence

                if class_confidence < threshold:
                    continue

                predictions.append({'xmin': x - w / 2,
                                    'ymin': y - h / 2,
                                    'xmax': x + w / 2,
                                    'ymax': h + h / 2,
                                    'confidence': class_confidence,
                                    'class_id': detected_class})

    predictions.sort(key=lambda a: a['confidence'], reverse=True)

    for i in range(len(predictions)):
        if predictions[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(predictions)):
            if intersection_over_union(predictions[i], predictions[j]) > 0.5:
                predictions[j]['confidence'] = 0
    final_objects = []
    for obj in predictions:
        # Validation bbox of detected object
        if obj['xmax'] > 416 or obj['ymax'] > 416 or obj['xmin'] < 0 or obj['ymin'] < 0 or obj['confidence'] <= 0:
            continue
        else:
            final_objects.append(
                (obj['confidence'], obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], obj['class_id']))

    if not final_objects:
        final_objects.append((-1, -1, -1, -1, -1, -1))

    return final_objects


def compute_entry(channel, cy, cx):
    return channel * YOLO_SIDE * YOLO_SIDE + cy * YOLO_SIDE + cx


def sigmoid(value: float):
    return 1 / (1 + math.exp(-value))


def softmax(values: list):
    temp = [math.exp(v) for v in values]
    total = sum(temp)
    return [t / total for t in temp]


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


lst = compute_prediction(result_list)

print()
