import cv2
import numpy as np
import os
import math


def preprocess(gray, img_width, img_height, is_vertical=False, is_reverse=False):
    # 1.Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
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
    blur = cv2.medianBlur(gray, 9)
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

    openning = cv2.morphologyEx(binary, cv2.MORPH_OPEN, dilation_y if is_vertical else dilation_x)
    # 4.膨胀一次，让轮廓突出
    # dilation = cv2.dilate(binary, dilation1_element, iterations=1)

    # 5.腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    # erosion = cv2.erode(dilation, erosion_element, iterations=1)

    # 6.再次膨胀，让轮廓明显一些
    # dilation2 = cv2.dilate(erosion, dilation2_element, iterations=3)

    # 7.存储中间图片
    cv2.imwrite('tmp/11sobel.png', gray)
    cv2.imwrite('tmp/12binary.png', binary)
    cv2.imwrite('tmp/13dilation.png', openning)
    # cv2.imwrite('tmp/14erosion.png', erosion)
    # cv2.imwrite('tmp/15dilation2.png', dilation2)
    # 最终使用的处理图片
    return openning


def findTextRegion(img):
    region = []
    # 1.查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 2.筛选那些面积小的
    for cnt in contours:
        # 计算该轮廓的面积
        # area = cv2.contourArea(cnt)
        # 面积小的都筛选掉
        # if (area < 200):
        #     continue
        # 获取外轮廓坐标
        box = cv2.boundingRect(cnt)
        region.append(box)
    return region


def detect(img):
    sp = img.shape
    img_height = sp[0]
    img_width = sp[1]
    print('img_width: ' + str(img_width))
    print('img_height: ' + str(img_height))
    # 1.转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2.形态学变换的预处理，得到可以查找矩形的图片
    # 预设图片竖向反转
    dilation_y_reverse = preprocess(gray, img_width, img_height, True, True)
    # 预设图片竖向不反转
    dilation_y = preprocess(gray, img_width, img_height, True, False)
    # 预设图片横向反转
    dilation_x_reverse = preprocess(gray, img_width, img_height, False, True)
    # 预设图片横向不反转
    dilation_x = preprocess(gray, img_width, img_height, False, False)

    # 3.查找和筛选文字区域
    region_x = findTextRegion(dilation_x)
    region_x_reverse = findTextRegion(dilation_x_reverse)
    region_y = findTextRegion(dilation_y)
    region_y_reverse = findTextRegion(dilation_y_reverse)

    # 4.用绿线画出这些找到的轮廓
    # 高度大于图片高度的0.3 或 小于图片高度的0.02，则筛除
    exclude_max_rate = 0.3
    exclude_min_rate = 0.02
    # 文本行宽高比
    text_rate = 0.5

    # 竖向查找
    max_w = 0
    min_w = 0
    array_w = []
    region_y_all = []
    true_region_w = 0
    # 筛选符合条件的轮廓，计算平均高度，高度标准差
    for x, y, w, h in region_y:
        # 高度大于图片高度的0.3 或 小于图片高度的0.02，则筛除
        if w > img_width * exclude_max_rate or w < img_width * exclude_min_rate:
            continue
        if w > max_w:
            max_w = w
        if min_w == 0 or w < min_w:
            min_w = w
        array_w.append(w)
        region_y_all.append((x, y, w, h))
    # 去除最大值最小值
    if max_w in array_w:
        array_w.remove(max_w)
    if min_w in array_w:
        array_w.remove(min_w)
    max_w = 0
    min_w = 0
    for x, y, w, h in region_y_reverse:
        # 高度大于图片高度的0.3 或 小于图片高度的0.02，则筛除
        if w > img_width * exclude_max_rate or w < img_width * exclude_min_rate:
            continue
        if w > max_w:
            max_w = w
        if min_w == 0 or w < min_w:
            min_w = w
        array_w.append(w)
        region_y_all.append((x, y, w, h))
    # 去除最大值最小值
    if max_w in array_w:
        array_w.remove(max_w)
    if min_w in array_w:
        array_w.remove(min_w)
    # 在原图画出轮廓，计算符合条件的文字行
    if len(array_w) > 0:
        # 计算平均高度，高度标准差
        avg_w = np.mean(np.array(array_w))
        print('avg_w: ' + str(avg_w))
        sd_w = np.sqrt(np.var(np.array(array_w)))
        print('sd_w: ' + str(sd_w))
        for x, y, w, h in region_y_all:
            # 高度小于平均高度的0.5，则筛除
            # if w < avg_w * 0.5:
            #     continue
            # 高度和平均高度差值在高度标准差以内则画出轮廓
            if abs(w - avg_w) <= sd_w:
                # print(f'{w}, {h} === {w / h}')
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 高宽比小于0.5的则认为是文本行
                if w / h < text_rate:
                    true_region_w += 1

    # 横向查找
    max_h = 0
    min_h = 0
    array_h = []
    region_x_all = []
    true_region_h = 0
    # 筛选符合条件的轮廓，计算平均高度，高度标准差
    for x, y, w, h in region_x:
        # 高度大于图片高度的0.3 或 小于图片高度的0.02，则筛除
        if h > img_height * exclude_max_rate or h < img_height * exclude_min_rate:
            continue
        if h > max_h:
            max_h = h
        if min_h == 0 or h < min_h:
            min_h = h
        array_h.append(h)
        region_x_all.append((x, y, w, h))
    # 去除最大值最小值
    if max_h in array_h:
        array_h.remove(max_h)
    if min_h in array_h:
        array_h.remove(min_h)
    max_h = 0
    min_h = 0
    for x, y, w, h in region_x_reverse:
        # 高度大于图片高度的0.3 或 小于图片高度的0.02，则筛除
        if h > img_height * exclude_max_rate or h < img_height * exclude_min_rate:
            continue
        if h > max_h:
            max_h = h
        if min_h == 0 or h < min_h:
            min_h = h
        array_h.append(h)
        region_x_all.append((x, y, w, h))
    # 去除最大值最小值
    if max_h in array_h:
        array_h.remove(max_h)
    if min_h in array_h:
        array_h.remove(min_h)
    # 在原图画出轮廓，计算符合条件的文字行
    if len(array_h) > 0:
        # 计算平均高度，高度标准差
        avg_h = np.mean(np.array(array_h))
        print('avg_h: ' + str(avg_h))
        sd_h = np.sqrt(np.var(np.array(array_h)))
        print('sd_h: ' + str(sd_h))
        for x, y, w, h in region_x_all:
            # 高度小于平均高度的0.5，则筛除
            # if h < avg_h * 0.5:
            #     continue
            # 高度和平均高度差值在高度标准差以内则画出轮廓
            if abs(h - avg_h) <= sd_h:
                # print(f'{w}, {h} === {h / w}')
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # 高宽比小于0.5的则认为是文本行
                if h / w < text_rate:
                    true_region_h += 1

    # cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('img', img)
    # 保存带轮廓的图片
    cv2.imwrite('tmp/16contours.png', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print('图片文字方向：')
    print('横向' if true_region_h >= true_region_w else '竖向')
    return 1 if true_region_h > true_region_w else 0


def single_test():
    # root = 'C:\\Users\\liusa\\Desktop\\202100583cy2bq.jpg'
    root = 'D:\\PycharmProjects\\vgg16\\data\\up\\202101209cybq.jpg'
    img = cv2.imread(root)
    res = detect(img)


if __name__ == '__main__':
    single_test()
    exit(0)
    from tqdm import tqdm
    root = 'D:\\PycharmProjects\\vgg16\\data\\up'
    res = 0
    count = 0
    for name in tqdm(sorted(os.listdir(os.path.join(root)))):
        print(os.path.join(root, name))
        img = cv2.imread(os.path.join(root, name))
        res += detect(img)
        count += 1
    print(str(res / count))
    pass
