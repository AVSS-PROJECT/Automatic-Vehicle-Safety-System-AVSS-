import cv2
from ultralytics import YOLO
import supervision as sv

frame_width = 1280
frame_height = 720

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("best.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        
        high_conf_detections = [
            (x, confidence, class_id, class_name) 
            for x, confidence, class_id, class_name 
            in detections if confidence > 0.7 
        ]

        labels = [
            f"{class_name} {confidence:0.2f}"
            for _, confidence, _, class_name 
            in high_conf_detections
        ]

        print("High confidence detections:", high_conf_detections)

        frame = box_annotator.annotate(
            scene=frame, 
            detections=high_conf_detections, 

        )

        cv2.imshow("yolov8", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
