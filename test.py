import cv2
from my_deepsort import MyDeepSort
from my_detect import MyDetect
from yolov5.utils.general import xywh2xyxy
from yolov5.utils.plots import plot_one_box

if __name__ == '__main__':
    det = MyDetect()
    deepsort = MyDeepSort()
    cap = cv2.VideoCapture('G:/repository/Yolov5_DeepSort_Pytorch/basketball.mp4')
    while True:
        _, image = cap.read()
        if image is None:
            break
        xywhs, confs, clss = det.detect(image)
        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, image)
        if len(outputs):
            for j, (output, conf) in enumerate(zip(outputs, confs)):
                bboxes = output[0:4]
                id = output[4]
                cls = output[5]
                # xyxys = xywh2xyxy(xywhs)
                # for xyxy, conf, cls in zip(xyxys, confs, clss):
                c = int(cls)
                # print(det.names[int(cls)])
                # print(conf.item())
                label = f'{id}{det.names[c]}{conf:.2f}'
                plot_one_box(bboxes, image, label=label, color=det.colors[c], line_thickness=2)
            cv2.imshow("test", image)
            if cv2.waitKey(1) == ord('q'):
                break
