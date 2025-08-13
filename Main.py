# ---- Qt GUI 오류 회피: pyplot import 전에 백엔드/환경 설정 ----
import os
os.environ.pop("QT_PLUGIN_PATH", None)  # 잘못된 Qt 경로 제거
import matplotlib
matplotlib.use("TkAgg")                 # Qt 대신 Tk 사용 (창 표시용)
# ----------------------------------------------------------------

import cv2
import matplotlib.pyplot as plt
import time

labels_to_names = {
    1:'person',2:'bicycle',3:'car',4:'motorcycle',5:'airplane',6:'bus',7:'train',8:'truck',9:'boat',10:'traffic light',
    11:'fire hydrant',12:'street sign',13:'stop sign',14:'parking meter',15:'bench',16:'bird',17:'cat',18:'dog',19:'horse',20:'sheep',
    21:'cow',22:'elephant',23:'bear',24:'zebra',25:'giraffe',26:'hat',27:'backpack',28:'umbrella',29:'shoe',30:'eye glasses',
    31:'handbag',32:'tie',33:'suitcase',34:'frisbee',35:'skis',36:'snowboard',37:'sports ball',38:'kite',39:'baseball bat',40:'baseball glove',
    41:'skateboard',42:'surfboard',43:'tennis racket',44:'bottle',45:'plate',46:'wine glass',47:'cup',48:'fork',49:'knife',50:'spoon',
    51:'bowl',52:'banana',53:'apple',54:'sandwich',55:'orange',56:'broccoli',57:'carrot',58:'hot dog',59:'pizza',60:'donut',
    61:'cake',62:'chair',63:'couch',64:'potted plant',65:'bed',66:'mirror',67:'dining table',68:'window',69:'desk',70:'toilet',
    71:'door',72:'tv',73:'laptop',74:'mouse',75:'remote',76:'keyboard',77:'cell phone',78:'microwave',79:'oven',80:'toaster',
    81:'sink',82:'refrigerator',83:'blender',84:'book',85:'clock',86:'vase',87:'scissors',88:'teddy bear',89:'hair drier',90:'toothbrush',
    91:'hair brush'
}

def main():
    cv_net_m = cv2.dnn_DetectionModel(
        r"D:\TOYPROJECT\ObjectDetection_SSD\pretrained\ssd_mobilenet_v3_large_coco_2020_01_14\frozen_inference_graph.pb",
        r"D:\TOYPROJECT\ObjectDetection_SSD\pretrained\ssd_config_02.pbtxt",
    )
    cv_net_m.setInputSize(320, 320)
    cv_net_m.setInputScale(1.0 / 127.5)
    cv_net_m.setInputMean((127.5, 127.5, 127.5))
    cv_net_m.setInputSwapRB(True)

    img = cv2.imread(r"D:\TOYPROJECT\ObjectDetection_SSD\images\EPL01.jpg")
    if img is None:
        raise FileNotFoundError("입력 이미지 경로를 확인하세요.")

    draw_img = img.copy()

    start_time = time.time()
    classIds, confs, bbox = cv_net_m.detect(img, confThreshold=0.5)
    end_time = time.time()
    detect_time_ms = (end_time - start_time) * 1000
    print(f"[INFO] Detect time: {detect_time_ms:.2f} ms")

    green = (0, 255, 0)
    red = (0, 0, 255)

    if bbox is not None and len(bbox) > 0:
        for class_id, confidence_score, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if confidence_score > 0.5:
                name = labels_to_names.get(int(class_id), str(int(class_id)))
                caption = f"{name}: {confidence_score:.4f}"
                x, y, w, h = map(int, box)
                cv2.rectangle(draw_img, (x, y), (x + w, y + h), color=green, thickness=2)
                cv2.putText(draw_img, caption, (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
                # print(caption, class_id, (x, y, w, h))

    # Matplotlib로 표시 (창 유지)
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 12))
    plt.imshow(draw_img)
    plt.axis("off")
    plt.tight_layout()

    plt.show(block=False)
    print("창을 닫거나 키/마우스를 누르면 종료됩니다.")
    plt.waitforbuttonpress(0)   # 무한 대기 (키/마우스 입력 시 반환)

if __name__ == "__main__":
    main()
