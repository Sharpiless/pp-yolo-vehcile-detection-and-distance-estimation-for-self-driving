from model import VehicleDetector
import cv2
from time import time
import imutils

if __name__ == '__main__':

    cap = cv2.VideoCapture('testttt.mp4')

    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    fps = int(cap.get(5))
    det = VehicleDetector()
    size = None
    print(fps)

    y1, x1, y2, x2 = 285, 628, 709, 1710

    while True:

        _, im = cap.read()

        if im is None:
            break
        
        im = imutils.resize(im[y1:y2, x1:x2], width=600)

        raw = im.copy()
        start = time()
        result = det.detect(im)['frame']
        if det.process:
            text = '{:.4f}'.format((time()-start)/2)
        cv2.putText(result, text, (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 4)
        if size is None:
            size = (result.shape[1], result.shape[0])
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result_car.mp4', fourcc, fps, size)
        videoWriter.write(result)
        cv2.imshow('a', result)

        if cv2.waitKey(1) == 27:
            break
        if cv2.getWindowProperty('a', cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
