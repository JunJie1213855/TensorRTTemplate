import numpy as np
from TRTInfer import preprocess,draw_detections,postprocess,TRTInference
import argparse
import cv2

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-trt', '--trt-file',default= "./yolov8n.engine",type=str,help="the path of trt engine")
    parser.add_argument('-f', '--im-file', default="./image/bus.jpg",type=str,help="the path of image")
    parser.add_argument("-s","--img-size",default=(640,480),nargs=2,help="the size of input image")
    args = parser.parse_args()
    m = TRTInference(args.trt_file)
    # 读取数据
    img = cv2.imread(args.im_file)
    img_input = preprocess(img,args.img_size)
    blob = {
        "images" : img_input
    }
    # 执行
    output = m(blob)
    img_output = postprocess(img,output["output0"],img.shape[:2],args.img_size)
    cv2.imwrite("./image/result.jpg",img_output)
    cv2.imshow("result",img_output)
    cv2.waitKey()
    # 测速
    print(f"one batch cost time : {m.speed(blob,100):.4f} s")