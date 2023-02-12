import cv2
from my_deepsort import MyDeepSort
from my_detect import MyDetect
from yolov5.utils.general import xywh2xyxy
from yolov5.utils.plots import plot_one_box

if __name__ == '__main__':
    det = MyDetect()
    deepsort = MyDeepSort()
    cap = cv2.VideoCapture(4)   #打开默认相机
    while True:
        _, image = cap.read()
        if image is None:
            break
        xywhs, confs, clss = det.detect(image)

        if xywhs is None or min(xywhs.shape) == 0 or min(confs.shape) == 0 or min(clss.shape) == 0:
            pass
        else:
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
        cv2.imshow("test", image)   #显示图像
        if cv2.waitKey(1) == ord('q'):  #
            break
