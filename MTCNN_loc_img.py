import cv2
from numpy.lib.twodim_base import eye
import tensorflow
import detect_face
import numpy as np
import sys
import os

#----tensorflow version check
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
print("Tensorflow version: ",tf.__version__)


def face_detection_MTCNN(detect_multiple_faces=False):
    #----var
    no_face_str = "No faces detected"

    #----MTCNN init
    color = (0,255,0)
    minsize = 4  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    with tf.Graph().as_default():
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,  # Cho phép chuyển đổi tự động sang thiết bị được hỗ trợ khi không tìm thấy thiết bị 
                                )
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        sess = tf.Session(config=config)
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    img = cv2.imread('32.png',1)
    img = cv2.resize(img,(720,480))
    # img = cv2.flip(img, 1)
    #----image processing
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    print("\nimage shape = ",img_rgb.shape)


    bounding_boxes, points = detect_face.detect_face(img_rgb, minsize, pnet, rnet, onet, threshold, factor) # tra ve toa do khung, tra ve 5 diem toa do mat mui mieng
    print("bounding_boxes = ",bounding_boxes[:])

    #----bounding boxes processing
    nrof_faces = bounding_boxes.shape[0]        # numberes of face: so luong khuon mat phat hien duoc
    print("nrof_faces=",nrof_faces)
    if nrof_faces > 0:                          # neu so luong khuon mat > 0
        points = np.array(points)               # ma tran cot
        points = np.transpose(points, [1, 0])   # dua ve ma tran hang
        points = points.astype(np.int16)        # dua ve dang so nguyen
        print("points==============",points)

        det = bounding_boxes[:, 0:4]            # lay tat ca cac hang va 4 cot la toa do 2 diem x1,y1,x2,y2 khung bounding_box    
        print("Do chinh xac {}".format(bounding_boxes[:,4]))
        print("''det'' =",det)
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]   # KQ: [480 640]
        if nrof_faces > 1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    # a = np.squeeze(det[i])            # in ra 2 toa do 2 nguoi cua 2 diem x1,y1,x2,y2 khung bounding_box
                    det_arr.append(np.squeeze(det[i]))  # them 2 toa do 2 nguoi cua 2 diem x1,y1,x2,y2 khung bounding_box
                    print("Yessssssss")

            else:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])   # tinh dien tich bounding_box (dai x rong)
                # print("bounding_box_size {} * {} = {}".format((det[:, 2] - det[:, 0]), (det[:, 3] - det[:, 1]),bounding_box_size ))
                img_center = img_size / 2               # KQ: [240 320]
                # print(img_center)
                offsets = np.vstack(                    # np.vstack : dua ma tran hang thanh ma tran cot
                    [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                # print('offsets=',[(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                # print(offsets)
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)                 # np.power: ham mu~ binh phuong
                # print(offset_dist_squared)
                index = np.argmax(
                    bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                # print("index = ",bounding_box_size - offset_dist_squared * 2.0)
                det_arr.append(det[index, :])
                # print(det[index, :])
                
        else:
            det_arr.append(np.squeeze(det))     # them toa do khung bounding_box vao "det_arr"
            print("det_arr = ''det'' = ",det_arr )

        det_arr = np.array(det_arr)             # lay phan tu mang
        print("det_arr = ''det'' = ",det_arr )
        det_arr = det_arr.astype(np.int16)      # dua ve so nguyen


        for i, det in enumerate(det_arr):
            #det = det.astype(np.int32)
            cv2.rectangle(img, (det[0],det[1]), (det[2],det[3]), color, 2)
            # print("i = ",index)
            # print("det = ",det)

            #----draw 5 point on the face
            # facial_points = points[index]       ###### chu y 
            facial_points = points[i]             ###### chu y
            print("facial_points = ",facial_points)     # points[i] = [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]
            for j in range(0,5,1):
                #cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
                cv2.circle(img, (facial_points[j], facial_points[j + 5]), 2, (0, 0, 255), -1, 1)
                if j == 2:
                    print(facial_points[j],"----",facial_points[j + 5] )


        cv2.line(img, (facial_points[2],0), (facial_points[2],facial_points[7]+200), color, 2)
        cv2.line(img, (0,facial_points[7]), (facial_points[2] +200,facial_points[7]), color, 2)            #

        x_eyes_left = facial_points[2] - facial_points[0]
        x_eye_right = facial_points[1] - facial_points[2]
        # print("x_eyes_left = {}---- x_eye_right = {} ".format(x_eyes_left, x_eye_right))

        y_eyes_left = facial_points[7] - facial_points[5]
        y_eye_right = facial_points[7] - facial_points[6]

        sai_so_x = x_eyes_left - x_eye_right
        print("sai_so_x = ",sai_so_x)
        sai_so_y = y_eyes_left - y_eye_right
        print("sai_so_y = ",sai_so_y)

        if abs(sai_so_x) <= 5:
            if abs(sai_so_y) <= 5:
                text = "Anh Dat "
                cv2.putText(img,  "{} X={:.2f} y={:.2f}".format(text, sai_so_x,sai_so_y), (int(det[0]), int(det[1]-5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2, lineType=1)
            else:
                text = "Chua Dat"
                cv2.putText(img,  "{} X={:.2f} y={:.2f}".format(text, sai_so_x,sai_so_y), (int(det[0]), int(det[1]-5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, lineType=1)
        
        else:
            text = "Chua Dat"
            cv2.putText(img,  "{} X={:.2f} y={:.2f}".format(text, sai_so_x,sai_so_y), (int(det[0]), int(det[1]-5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, lineType=1)


    # ----no faces detected
    else:
        cv2.putText(img, no_face_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    #----image display
    cv2.imshow("demo by JohnnyAI", img)

    #----'q' key pressed?
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit("Thanks")

    #----release
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_detection_MTCNN(detect_multiple_faces=False)
