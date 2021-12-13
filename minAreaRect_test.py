import cv2
import numpy as np
from rotate_img import rotate_bound
import math


def preprocess(gray, img_width, img_height, is_vertical=False, is_reverse=False):
    # 1.Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 简单阈值
    # if is_reverse:
    #     ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # else:
    #     ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # 高斯模糊
    # blur = cv2.GaussianBlur(sobel, (3, 3), 0)
    # 二值化
    # if is_reverse:
    #     ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # else:
    #     ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # 中值滤波
    blur = cv2.medianBlur(gray, 7)
    # 自适应阈值
    if is_reverse:
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
    else:
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)

    # 3.膨胀和腐蚀操作的核函数
    # 高度膨胀图片高度的0.002，宽度膨胀图片宽度的0.01
    dilation_x = (math.ceil(img_width * 0.01), math.ceil(img_width * 0.01 * 0.2))
    dilation_y = (math.ceil(img_height * 0.01 * 0.2), math.ceil(img_height * 0.01))
    dilation1_element = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_y if is_vertical else dilation_x)

    erosion_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation2_element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    # 4.膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, dilation1_element, iterations=1)

    # 5.腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    # erosion = cv2.erode(dilation, erosion_element, iterations=1)

    # 6.再次膨胀，让轮廓明显一些
    # dilation2 = cv2.dilate(erosion, dilation2_element, iterations=3)

    # 7.存储中间图片
    cv2.imwrite('tmp/11sobel.png', blur)
    cv2.imwrite('tmp/12binary.png', binary)
    cv2.imwrite('tmp/13dilation.png', dilation)
    # cv2.imwrite('tmp/14erosion.png', erosion)
    # cv2.imwrite('tmp/15dilation2.png', dilation2)
    # 最终使用的处理图片
    return dilation


def findTextRegion(img):
    region = []

    # 1.查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2.筛选那些面积小的
    for cnt in contours:
        # # 计算该轮廓的面积
        # area = cv2.contourArea(cnt)
        #
        # # 面积小的都筛选掉
        # if (area < 100):
        #     continue
        #
        # # 轮廓近似，作用很小
        # epsilon = 0.001 * cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        print('rect is: ')
        print(rect)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        # if (height > width * 1.2):
        #     continue

        region.append(box)
    return region


def rotate_img(img):
    region = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if rect[2] > 0 and rect[2] < 90:
            region.append(rect[2])
    avg_num = np.mean(np.array(region))
    sd = np.sqrt(np.var(np.array(region)))
    print('avg_num: ' + str(avg_num))
    print('sd: ' + str(sd))
    return abs(sd - avg_num)


def detect(img):
    sp = img.shape
    img_height = sp[0]
    img_width = sp[1]
    print('img_width: ' + str(img_width))
    print('img_height: ' + str(img_height))

    # 1.转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2.形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray, img_width, img_height, False, False)

    angle = rotate_img(dilation)
    print('角度：' + str(angle))

    # img = rotate_bound(img, angle)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilation = preprocess(gray, img_width, img_height, False, False)

    # 3.查找和筛选文字区域
    region = findTextRegion(dilation)

    # 4.用绿线画出这些找到的轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('img', img)

    # 带轮廓的图片
    cv2.imwrite('tmp/6contours.png', img)

    # cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    x = 'C:\\Users\\liusa\\Desktop\\2021824-145825.png'
    img = cv2.imread(x)
    detect(img)
    pass
