import cv2
import datetime, time
from pathlib import Path

from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

model = YOLO('yolov8n.pt')

def capture_and_save(frame):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img)

    for r in results:

        annotator = Annotator(frame)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

    frame = annotator.result()

    m = 0
    p = Path("images")
    for imp in p.iterdir():
        if imp.suffix == ".png" and imp.stem != "last":
            num = imp.stem.split("_")[1]
            try:
                num = int(num)
                if num > m:
                    m = num
            except:
                print("Error reading image number for", str(imp))
    m += 1
    lp = Path("images/last.png")
    if lp.exists() and lp.is_file():
        np = Path("images/img_{}.png".format(m))
        np.write_bytes(lp.read_bytes())
    cv2.imwrite("images/last.png", frame)


if __name__ == "__main__":
    capture_and_save()
    print("done")
