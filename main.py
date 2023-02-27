import json
import numpy as np
import cv2
import onnxruntime

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

num_classes = 5


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

                boxes.append(tc)

    return boxes


r = compute_bounding_boxes(result)

print()
