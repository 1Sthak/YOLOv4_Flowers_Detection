import cv2
import numpy as np
import tensorflow as tf

# gpu kontrolü
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# model dosyalarının değişkene eşitlenmesi
model = cv2.dnn.readNetFromDarknet("model/cicekler_yolov4.cfg", "model/cicekler_yolov4_last.weights")
#
layers = model.getLayerNames()
unconnect = model.getUnconnectedOutLayers()
unconnect = unconnect - 1
font = cv2.FONT_HERSHEY_SIMPLEX

output_layers = []
for i in unconnect:
    output_layers.append(layers[int(i)])

classFile = 'cicek.names'  # dikkat et
classNames = []

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# resimleri okuma
img = cv2.imread('images/ters-lalee.jpg')
# yeniden boyutlandırma
img = cv2.resize(img, (1600, 900))
# isimlendirilmiş pencere
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
# tam ekran olarak görüntüleme
cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# görüntü boyutunun değişkene eşitlenmesi
img_width = img.shape[1]
img_height = img.shape[0]

# resmi blob formata çevirme (blob: resmin 4 boyutluk tensörlere çevirilmiş halidir) bu işlem görüntüyü modele
# vermemiz için gereklidir
img_blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True)

model.setInput(img_blob)
detection_layers = model.forward(output_layers)

ids_list = []
boxes_list = []
confidences_list = []

for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]

        if confidence > 0.80:
            label = classNames[predicted_id]
            bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
            start_x = int(box_center_x - (box_width / 2))
            start_y = int(box_center_y - (box_height / 2))

            ids_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append([start_x, start_y, int(box_width), int(box_height)])

max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

if len(max_ids) > 0:
    for max_id in max_ids:
        max_class_id = max_id
        box = boxes_list[max_class_id]

        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_id = ids_list[max_class_id]
        confidence = confidences_list[max_class_id]

        label = classNames[predicted_id]

        end_x = start_x + box_width
        end_y = start_y + box_height
        cenx = start_x + ((end_x - start_x) / 2)
        ceny = start_y + ((end_y - start_y) / 2)
        alan = (end_x - start_x) * (end_y - start_y)
        frsize = img_height * img_width
        hedefBoyutu = (alan * 100) / frsize

        label = "{}: {:.2f}%".format(label, confidence * 100)
        cv2.putText(img, label, (start_x, start_y - 10), font, 1, (0, 0, 255), 2)

        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

else:
    print("Hiç nesne tespit edemedik")

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
