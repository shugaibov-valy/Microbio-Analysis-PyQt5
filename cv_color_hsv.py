#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np


def target_hsv(file_name):
    image = cv2.imread(f'{file_name}')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

   # plt.imshow(hsv);
    cropped = image[15:16, 570:571]

   # plt.imshow(cropped);
    hsv_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    return hsv_cropped[0][0]




def count_circles(file_name, hsv_):
    # Выбор загружаемой фотографии

    image = cv2.imread(f'{file_name}')

    original = image.copy()

    #  Кодировка картинки в HSV

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Выбор диапазонов цветов

    lower = np.array(hsv_)
    upper = np.array(hsv_)


    # Создание маски на основе диапазона цветов

    mask = cv2.inRange(hsv, lower, upper)
    x, y = mask.shape

    Big_mask = cv2.resize(mask, (x, y), cv2.INTER_LINEAR)

    canny = cv2.Canny(mask, 30, 150, 3)
    dilated = cv2.dilate(canny, (1, 1), iterations=2)


    count = 0
    for i in range(3):
        circles = cv2.HoughCircles(cv2.GaussianBlur(Big_mask, (21, 21), cv2.BORDER_DEFAULT), cv2.HOUGH_GRADIENT, 0.9,
                                   120, param1=50, param2=30, minRadius=0, maxRadius=0)
        detected = np.uint16(np.around(circles))

        count = 0
        for (x, y, r) in detected[0, :]:
            cv2.circle(Big_mask, (x, y), r, (0, 0, 0), 3)
            count += 1

        w, h = Big_mask.shape
        Big_mask = cv2.resize(Big_mask, (w * 2, h * 2), cv2.INTER_LINEAR)

    (cnt, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(original, cnt, -1, (255, 255, 255), 2)
    cv2.imwrite('original.png', original)


    return count