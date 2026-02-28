import cv2
import numpy as np
from typing import Tuple

# 绘制定位框
def draw_detections(img: np.ndarray, box: list[float] | tuple[float, float, float, float], score: float, class_id: int) -> None:
        """
        绘制定位框，包括类别，类别得分数。
        Args:
            img: 输入图像
            box: 定位框的信息
            score: 定位框分数（置信度）
            class_id: 定位框的目标类别
        Returns:
            None
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box
        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 2)
        # Draw the label text on the image
        cv2.putText(img, f"id:{class_id},score:{score:.3f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
# 预处理
def preprocess(img: np.ndarray, size: tuple[int, int] | list[int]) -> np.ndarray:
    """
    图像预处理
    Args:
	    img : 输入图像数据
	    size: 模型输入的尺寸
    Returns:
	    预处理图像结果
    """
    # 获取原图像的尺寸
    img_height, img_width = img.shape[:2]
    # 将图像的空间 BGR 转换为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 获取尺寸
    input_height, input_width = size
    # 调整图像的尺寸
    img = cv2.resize(img, (input_width, input_height))
    #归一化数据 [0,255] -> [0,1]
    image_data = np.array(img) / 255.0
    # 数据从 HWC 转换为 CHW
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
    # 拓展维度 1 X C X H X W
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    #转换为连续的数据
    image_data = np.ascontiguousarray(image_data, dtype=np.float32)
    # 返回图像数据
    return image_data

# 后处理
def postprocess(input_image: np.ndarray, output: dict[str, np.ndarray], original_size: tuple[int, int] | list[int], model_size: tuple[int, int] | list[int]) -> np.ndarray:
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): 输入图像
            output (numpy.ndarray): 模型输出. (84,8400)
            original_size (tuple): 原始图像的尺寸
            model_size (tuple): 模型输入图像的尺寸
        Returns:
            numpy.ndarray: 带有定位框信息的图像
        Ps:
        the detailed format of output data
            output : 84 X 8400
            [
                [x,y,w,h,class1_score,class2_score,...,class80_score], # object 1st
                [x,y,w,h,class1_score,class2_score,...,class80_score], # object 2nd
                ... ,
                [x,y,w,h,class1_score,class2_score,...,class80_score], # object 8400th
            ]
        """

        # 转置 84 X 8400 --> 8400 X 84
        outputs = np.transpose(np.squeeze(output[0]))

        # 获取目标数 8400
        rows = outputs.shape[0]

        # 定位框、置信度、类别索引列表
        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []

        #x、y方向的尺度因子
        x_factor = original_size[1] / model_size[1]
        y_factor = original_size[0] / model_size[0]

        # 循环
        for i in range(rows):
            # 获取类别序列
            classes_scores = outputs[i][4:]

            # 找到最高分数值
            max_score = np.amax(classes_scores)

            # 当置信度大于阈值时
            if max_score >= 0.5:
                # 获取最高置信度的索引数
                class_id = int(np.argmax(classes_scores))

                # 输出定位框信息
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 获取左上、右下的点
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # 添加类别索引、最大置信度、定位框
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # 极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)
        # 索引迭代
        for i in indices:
            # 获取极大值抑制后定位框、置信度、类别
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # 绘制定位框
            draw_detections(input_image, box, score, class_id)

        # 返回目标定位后的图像
        return input_image