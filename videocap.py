import cv2
import numpy as np
from centroidtracker import CentroidTracker


cap = cv2.VideoCapture(0)
proto = r"./deploy.prototxt"
model = r"res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto, model)

while(True):
    ret, frame = cap.read()
    avg = cv2.blur(frame, (4, 4))
    avg_float = np.float32(avg)

    # blobFromImage 对图像进行预处理，包括减均值，比例缩放，裁剪，交换通道等，返回一个4通道的blob(blob可以简单理解为一个N维的数组，用于神经网络的输入)
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    rects = []
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.9:
            # draw the bbox
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            rects.append(box.astype("int"))
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)

    ct = CentroidTracker()
    objects = ct.update(rects)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
