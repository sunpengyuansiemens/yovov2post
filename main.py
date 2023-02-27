import json
import numpy as np
import cv2
import onnxruntime
import math

model = "./tinyyolov2-8.onnx"
path = "./test.JPG"

# Preprocess the image
img = cv2.imread(path)
img = cv2.resize(img, dsize=(416, 416), interpolation=cv2.INTER_AREA)
img.resize((1, 3, 416, 416))

data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

result = session.run([output_name], {input_name: data})

num_classes = 20

anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
           5.47434, 7.88282, 3.52778, 9.77052, 9.16828]


def compute_bounding_boxes(model_result):
    boxes = []

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
                x = cx + sigmoid(tx) * 32
                y = cy + sigmoid(ty) * 32
                w = math.exp(tw) * anchors[2*b] * 32
                h = math.exp(th) * anchors[2*b + 1] * 32
                confidence = sigmoid(tc)

                classes = [0 for i in range(num_classes)]
                for c in range(num_classes):
                    classes[c] = model_result[0][0][channel + 5 + c][cx][cy]

                detectedClass, bestClassScore = softmax(classes)

                confidenceInClass = bestClassScore * confidence

    return boxes


def softmax(classes):
    temp = np.exp(classes) / np.sum(np.exp(classes))
    return np.argmax(temp), np.max(temp)


def sigmoid(value:float):
    return 1/(1 + math.exp(-value))


r = compute_bounding_boxes(result)

print()
