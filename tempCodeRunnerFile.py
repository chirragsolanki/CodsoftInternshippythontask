import cv2
import numpy as np

# Define parameters for emotion detection
emotion_dict = {
    0: 'neutral', 
    1: 'happiness', 
    2: 'surprise', 
    3: 'sadness',
    4: 'anger', 
    5: 'disgust', 
    6: 'fear'
}

# Define parameters for gender and age detection
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male','Female']

# Initialize gender and age detection models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Initialize the emotion detection model
emotion_model = cv2.dnn.readNetFromONNX('emotion-ferplus-8.onnx')

# Initialize the object detection model
obj_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
obj_classes = []
with open("coco.names", "r") as f:
    obj_classes = f.read().strip().split("\n")

# Function to detect faces and objects and highlight them
def highlightFaceAndObjects(face_net, obj_net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    
    # Face detection
    blob_face = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob_face)
    detections_face = face_net.forward()
    face_boxes = []
    for i in range(detections_face.shape[2]):
        confidence = detections_face[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections_face[0, 0, i, 3] * frame_width)
            y1 = int(detections_face[0, 0, i, 4] * frame_height)
            x2 = int(detections_face[0, 0, i, 5] * frame_width)
            y2 = int(detections_face[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    
    # Object detection
    blob_obj = cv2.dnn.blobFromImage(frame_opencv_dnn, 1/255.0, (416, 416), swapRB=True, crop=False)
    obj_net.setInput(blob_obj)
    obj_outputs = obj_net.forward(get_outputs_names(obj_net))

    for output in obj_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame_opencv_dnn, (x, y), (x + w, y + h), (0, 255, 255), 2)
                label = f"{obj_classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame_opencv_dnn, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame_opencv_dnn, face_boxes

def get_outputs_names(net):
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Function to detect gender and age
def detectGenderAndAge(face, age_net, gender_net, age_list, gender_list):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    print(f'Gender: {gender}')

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]
    print(f'Age: {age[1:-1]} years')

    return gender, age

# Function to detect emotion
def detectEmotion(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resize_frame = cv2.resize(gray, (64, 64))
    resize_frame = resize_frame.reshape(1, 1, 64, 64)
    model.setInput(resize_frame)
    output = model.forward()

    # Get predicted emotion
    pred_emotion = emotion_dict[list(output[0]).index(max(output[0]))]
    return pred_emotion

# Main function for live webcam feed
def detectEmotionGenderAndAge(model):
    cap = cv2.VideoCapture('sem-8project.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result_img, face_boxes = highlightFaceAndObjects(faceNet, obj_net, frame)
        if not face_boxes:
            print("No face detected")
            continue

        for face_box in face_boxes:
            face = frame[max(0, face_box[1] - 20):min(face_box[3] + 20, frame.shape[0] - 1),
                         max(0, face_box[0] - 20):min(face_box[2] + 20, frame.shape[1] - 1)]
            gender, age = detectGenderAndAge(face, ageNet, genderNet, ageList, genderList)

            # Draw gender and age information on the frame
            cv2.putText(result_img, f'{gender}, {age}', (face_box[0], face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # Detect emotion
            pred_emotion = detectEmotion(face, model)

            cv2.putText(
                result_img,
                pred_emotion,
                (face_box[0], face_box[3] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (215, 5, 247),
                2,
                lineType=cv2.LINE_AA
            )

        cv2.imshow("Detecting age, gender, and emotion", result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the combined detection function
detectEmotionGenderAndAge(emotion_model)
