# 1. 라이브러리 불러오기
from ultralytics import YOLO

# 2. 추론기 초기화
model = YOLO("yolo26n.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# 3. 이미지 불러오기
source = "https://ultralytics.com/images/bus.jpg"

# 4. 추론하기
results = model(source)  # predict on an image

# 5. 시각화하기
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk