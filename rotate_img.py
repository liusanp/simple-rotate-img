import cv2
import numpy as np
import math
from logger import logger


# 旋转图片
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


# 计算方位角函数，线段与x轴的夹角
def azimuth_angle(x1, y1, x2, y2):
    a = math.atan2((y1 - y2), (x2 - x1))
    return (a * 180 / math.pi)


# 计算两点间长度
def get_length(x1, y1, x2, y2):
    # 用math.sqrt（）求平方根
    len = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
    return len


# 计算旋转角度平均值，传入角度集合，返回角度平均值
def cal_angle(angles):
    avg_num = np.mean(np.array(angles))
    sd = np.sqrt(np.var(np.array(angles)))
    logger.info('avg_num: ' + str(avg_num))
    logger.info('sd: ' + str(sd))
    return avg_num


if __name__ == '__main__':
    image = cv2.imread('C:\\Users\\liusa\\Desktop\\202199-95311.png')
    angle = 39
    img = rotate_bound(image, angle)
    cv2.imshow('ww', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass
