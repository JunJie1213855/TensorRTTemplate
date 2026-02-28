import numpy as np
from TRTInfer import preprocess, draw_detections, postprocess, TRTInference
import argparse
import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-trt', '--trt-file', default="./yolov8n.engine", type=str, help="the path of trt engine")
    parser.add_argument('-f', '--im-file', default="./image/bus.jpg", type=str, help="the path of image")
    parser.add_argument("-s", "--img-size", default=(640, 480), nargs=2, type=int, help="the size of input image")
    args = parser.parse_args()

    m: TRTInference = TRTInference(args.trt_file)

    # 读取数据
    img: np.ndarray = cv2.imread(args.im_file)
    if img is None:
        raise FileNotFoundError(f"Image file not found: {args.im_file}")

    img_input: np.ndarray = preprocess(img, tuple(args.img_size))
    blob: dict[str, np.ndarray] = {
        "images": img_input
    }

    # 执行
    output: dict[str, np.ndarray] = m(blob)
    img_output: np.ndarray = postprocess(img, output["output0"], img.shape[:2], tuple(args.img_size))

    cv2.imwrite("./image/result.jpg", img_output)
    cv2.imshow("result", img_output)
    cv2.waitKey()

    # 测速
    avg_time: float = m.speed(blob, 100)
    print(f"one batch cost time : {avg_time:.4f} s")


if __name__ == "__main__":
    main()