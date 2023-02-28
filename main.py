import json
import numpy as np
import cv2
import onnxruntime
import math
import copy

model = "./tinyyolov2-8.onnx"
path = "./test.JPG"

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

num_classes = 20

anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
           5.47434, 7.88282, 3.52778, 9.77052, 9.16828]


class Prediction:
    def __init__(self, x, y, w, h, c, i):
        self.x = x - w / 2
        self.y = y - h / 2
        self.width = w
        self.height = h
        self.confidence = c
        self.index = i

    def getConfidence(self):
        return self.confidence


def compute_prediction_list(model_result):
    predictions = []

    for cy in range(13):
        for cx in range(13):
            for b in range(5):
                channel = b * (num_classes + 5)
                tx = model_result[0][0][channel][cy][cx]
                ty = model_result[0][0][channel + 1][cy][cx]
                tw = model_result[0][0][channel + 2][cy][cx]
                th = model_result[0][0][channel + 3][cy][cx]
                tc = model_result[0][0][channel + 4][cy][cx]

                # apply sigmoid function to raw data
                x = (cx + sigmoid(tx)) * 32
                y = (cy + sigmoid(ty)) * 32
                w = math.exp(tw) * anchors[2 * b] * 32
                h = math.exp(th) * anchors[2 * b + 1] * 32
                confidence = sigmoid(tc)

                classes = [0 for i in range(num_classes)]
                for c in range(num_classes):
                    classes[c] = model_result[0][0][channel + 5 + c][cx][cy]

                classes = softmax(classes)

                originalClass = copy.deepcopy(classes)
                bestClassScore = getMax(classes)
                detectedClass = originalClass.index(bestClassScore)
                confidenceInClass = bestClassScore * confidence
                if confidenceInClass < 0.3:
                    continue
                prediction = Prediction(x, y, w, h, confidenceInClass, detectedClass)
                predictions.append(prediction)

    return predictions


def compute_prediction(model_result):
    predictions = []

    for cy in range(13):
        for cx in range(13):
            for b in range(5):
                channel = b * (num_classes + 5)
                tx = model_result[compute_entry(channel, cy, cx)]
                ty = model_result[compute_entry(channel + 1, cy, cx)]
                tw = model_result[compute_entry(channel + 2, cy, cx)]
                th = model_result[compute_entry(channel + 3, cy, cx)]
                tc = model_result[compute_entry(channel + 4, cy, cx)]

                # apply sigmoid function to raw data
                x = (cx + sigmoid(tx)) * 32
                y = (cy + sigmoid(ty)) * 32
                w = math.exp(tw) * anchors[2 * b] * 32
                h = math.exp(th) * anchors[2 * b + 1] * 32
                confidence = sigmoid(tc)

                classes = [0 for i in range(num_classes)]
                for c in range(num_classes):
                    classes[c] = model_result[compute_entry(channel + 5 + c, cx, cy)]

                classes = softmax(classes)

                originalClass = copy.deepcopy(classes)
                bestClassScore = getMax(classes)
                detectedClass = originalClass.index(bestClassScore)
                confidenceInClass = bestClassScore * confidence
                if confidenceInClass < 0.3:
                    continue
                prediction = Prediction(x, y, w, h, confidenceInClass, detectedClass)
                predictions.append(prediction)

    return predictions


def compute_entry(channel, cy, cx):
    return channel * 13 * 13 + cy * 13 + cx


def getMax(classes):
    classes.sort(reverse=True)
    return classes[0]


def sigmoid(value: float):
    return 1 / (1 + math.exp(-value))


def softmax(values: list):
    temp = [math.exp(v) for v in values]
    total = sum(temp)
    return [t / total for t in temp]


lst = compute_prediction_list(result)
lst1 = compute_prediction(result_list)

lst.sort(key=lambda a: a.getConfidence(), reverse=True)
lst1.sort(key=lambda a: a.getConfidence(), reverse=True)

print()
