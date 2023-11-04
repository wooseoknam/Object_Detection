# # # import cv2
# # # import numpy as np

# # # img = cv2.imread('/Users/wooseoknam/Desktop/red.png')
# # # print(img[0].shape)



# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # img = cv2.imread('/Users/wooseoknam/Desktop/Object_Detection/YOLOX/YOLOX_outputs/yolox_s/vis_res/2023_04_18_14_43_33/car_90.jpg', cv2.IMREAD_COLOR)
# img = cv2.imread('/Users/wooseoknam/Desktop/Object_Detection/crawled_img/car_83.jpg', 1)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # ret, thresh1 = cv2.threshold(img,127,255, cv2.THRESH_BINARY)
# # ret, thresh2 = cv2.threshold(img,127,255, cv2.THRESH_BINARY_INV)
# # ret, thresh3 = cv2.threshold(img,127,255, cv2.THRESH_TRUNC)
# # ret, thresh4 = cv2.threshold(img,127,255, cv2.THRESH_TOZERO)
# # ret, thresh5 = cv2.threshold(img,127,255, cv2.THRESH_TOZERO_INV)
# plt.imshow(img)
# plt.xticks([]),plt.yticks([])

# plt.show()

# import numpy as np
# import matplotlib.cm as cm
# from matplotlib import pyplot as plt

# Z=np.array(
# [
#      [[160, 170, 180]]
# ])

# plt.imshow(Z)
# plt.show()


# import extcolors
# from PIL import Image


# img = Image.open('/Users/wooseoknam/Desktop/Object_Detection/crawled_img/car_78.jpg')
# colors, pixel_count = extcolors.extract_from_image(img)
# print(colors)
# pixel_output = 0
# for c in colors:
#     pixel_output += c[1]
#     print(f'{c[0]} : {round((c[1] / pixel_count) * 100, 2)}% ({c[1]})')
# print(f'Pixels in output: {pixel_output} of {pixel_count}')

# # import cv2
# # import numpy as np
# # from scipy.spatial import distance as dist



# # # Contour 영역 내에 텍스트 쓰기
# # # https://github.com/bsdnoobz/opencv-code/blob/master/shape-detect.cpp
# # def setLabel(image, str, contour):

# #    fontface = cv2.FONT_HERSHEY_SIMPLEX
# #    scale = 0.6
# #    thickness = 2

# #    size = cv2.getTextSize(str, fontface, scale, thickness)
# #    text_width = size[0][0]
# #    text_height = size[0][1]

# #    x, y, width, height = cv2.boundingRect(contour)

# #    pt = (x + int((width - text_width) / 2), y + int((height + text_height) / 2))
# #    cv2.putText(image, str, pt, fontface, scale, (255, 255, 255), thickness, 8)



# # # 컨투어 내부의 색을 평균내서 red, green, blue 중 어느 색인지 체크 
# # def label(image, contour):


# #    mask = np.zeros(image.shape[:2], dtype="uint8")
# #    cv2.drawContours(mask, [contour], -1, 255, -1) 

# #    mask = cv2.erode(mask, None, iterations=2)
# #    mean = cv2.mean(image, mask=mask)[:3] 


# #    minDist = (np.inf, None)



# #    for (i, row) in enumerate(lab):

# #        d = dist.euclidean(row[0], mean)

# #        if d < minDist[0]:
# #            minDist = (d, i)

# #    return colorNames[minDist[1]]



# # # 인식할 색 입력 
# # colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]
# # colorNames = ["red", "green", "blue"]



# # lab = np.zeros((len(colors), 1, 3), dtype="uint8")
# # for i in range(len(colors)):
# #    lab[i] = colors[i] 

# # lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)




# # # 원본 이미지 불러오기 
# # image = cv2.imread('/Users/wooseoknam/Desktop/temp2.jpeg', 1)


# # blurred = cv2.GaussianBlur(image, (5, 5), 0)

# # # 이진화 
# # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) 
# # ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

# # # 색검출할 색공간으로 LAB사용 
# # img_lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

# # thresh = cv2.erode(thresh, None, iterations=2) 
# # cv2.imshow("Thresh", thresh)


# # # 컨투어 검출
# # contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# # # 컨투어 리스트가 OpenCV 버전에 따라 차이있기 때문에 추가 
# # if len(contours) == 2:
# #    contours = contours[0]

# # elif len(contours) == 3:
# #    contours = contours[1]


# # # 컨투어 별로 체크 
# # for contour in contours:

# #    cv2.imshow("Image", image)
# #    cv2.waitKey(0)

# #    # 컨투어를 그림 
# #    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)


# #    # 컨투어 내부에 검출된 색을 표시 
# #    color_text = label(img_lab, contour)
# #    setLabel(image, color_text, contour)


# # cv2.imshow("Image", image)
# # cv2.waitKey(0)



# import numpy as np
# import cv2
# from scipy.spatial import distance as dist

# image = cv2.imread('/Users/wooseoknam/Desktop/blue.png', 1)
# colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]
# colorNames = ["red", "green", "blue"]

# mask = np.zeros(image.shape[:2], dtype="uint8")
# mask = cv2.erode(mask, None, iterations=2)

# mean = cv2.mean(image, mask=mask)[:3] 
# print(mean)

# minDist = (np.inf, None)

# lab = np.zeros((len(colors), 1, 3), dtype="uint8")

# for (i, row) in enumerate(lab):

#     d = dist.euclidean(row[0], mean)

#     if d < minDist[0]:
#         minDist = (d, i)

# print(minDist)
# print(colorNames[minDist[1]])







# from colorthief import ColorThief
# import matplotlib.pyplot as plt
# import colorsys
# import cv2
# import extcolors
# from PIL import Image




# img = cv2.imread('/Users/wooseoknam/Desktop/Object_Detection/YOLOX/YOLOX_outputs/yolox_s/vis_res/2023_04_18_14_43_33/car_90.jpg')
# color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # # ct = ColorThief("/Users/wooseoknam/Desktop/red.png")
# # ct = ColorThief(img)
# # print(ct)
# # dominant_color = ct.get_color(quality=1)
# # print([[dominant_color]])
# # # plt.imshow([[dominant_color]])
# # # plt.show()

# pil_image=Image.fromarray(color_coverted)
# colors, pixel_count = extcolors.extract_from_image(pil_image)
# print(colors[0])



# import os

# # gpu_list = os.getenv('CUDA_VISIBLE_DEVICES', None)
# # print(gpu_list)


# devices_list_info = os.popen("nvidia-smi -L")
# print(devices_list_info)

# import glob
# from PIL import Image

# imglist = glob.glob("/Users/wooseoknam/Desktop/Object_Detection/crawled_img")
# print(imglist)

# for img_path in imglist:
#   	img = Image.open(img_path)
#     img.resize((640,640)).save(img_path) 
# import numpy as np

# nd3 = np.array([
#     [[1, 1, 2], [2, 1, 2], [3, 1, 2], [4, 1, 2], [5, 1, 2]],
#     [[6, 1, 2], [7, 1, 2], [8, 1, 2], [9, 1, 2], [10, 1, 2]],
#     [[11, 1, 2], [12, 1, 2], [13, 1, 2], [14, 1, 2], [15, 1, 2]],
#     [[16, 1, 2], [17, 1, 2], [18, 1, 2], [19, 1, 2], [20, 1, 2]],
#     ])
# print(nd3.shape)
# print((nd3.sum(axis=0)).sum(axis=0))
# print(nd3.sum(axis=1).sum(axis=1))
# print(nd3.sum(axis=2))

# for i in nd3:
#     for j in i:
#         print(j)

# N, M, K = map(int, input().split()) # 5, 8, 3

# a, b, c, d, e = map(int, input().split())

# lst = [a, b, c, d, e]# [2, 4, 5, 4, 6]
# lst.sort(reverse=1)
# x = 0
# count = 0
# idx = 0

# for i in range(M):
#     x += lst[idx] # [6, 5, 4, 4, 2]
#     count += 1
#     if count == K:
#         x -= lst[idx]
#         x += lst[idx + 1]
#         count = 0

# print(x)



# N, M = map(int, input().split())

# lst = []
# for i in range(N):
#     data = list(map(int, input().split()))
#     lst.append(data)

# _min = min(lst[0])
# row = 0

# for i in lst:
#     if min(i) > _min:
#         row = lst.index(i)

# print(min(lst[row]))



# N, K = map(int, input().split())
# cnt = 0

# while True:
#     if N % K != 0:
#         N -= 1
#         cnt += 1
#     elif N % K == 0:
#         N /= K
#         cnt += 1
#     if N == 1:
#         break

# print(cnt)




# import cv2
# import numpy as np

# #---① BGR 컬러 스페이스로 원색 픽셀 생성
# red_bgr = np.array([[[10,20,30]]], dtype=np.uint8)   # 빨강 값만 갖는 픽셀

# #---② BGR 컬러 스페이스를 HSV 컬러 스페이스로 변환
# red_hsv = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2HSV)

# #---③ HSV로 변환한 픽셀 출력
# print("red:",red_hsv)

# import numpy as np

# hsv_color = np.array([
#     [[1, 1, 2], [2, 1, 2], [3, 1, 2], [4, 1, 2], [5, 1, 2]],
#     [[6, 1, 2], [7, 1, 2], [8, 1, 2], [9, 1, 2], [10, 1, 2]],
#     [[11, 1, 2], [12, 1, 2], [13, 1, 2], [14, 1, 2], [15, 1, 2]],
#     [[16, 1, 2], [17, 1, 2], [18, 1, 2], [19, 1, 2], [20, 1, 2]],
#     ])

# v = ((hsv_color.sum(axis=0)).sum(axis=0)[2]) / (hsv_color.shape[0] * hsv_color.shape[1])
# print(hsv_color.sum(axis=0))
# print((hsv_color.sum(axis=0)).sum(axis=0)[2])
# print(v)






# equation = input()
# if '-' not in equation:
#     equation = equation.split('+')
#     lst = []
#     for i in equation:
#         i = int(i)
#         lst.append(i)
#     print(sum(lst))
# elif '-' in equation:
#     equation = equation.split('-')
#     m = int(equation[0])
#     for i in equation[1:]:
#         s = 0
#         if '+' not in i:
#             m -= int(i)
#         elif '+' in i:
#             i = i.split('+')
#             for j in i:
#                 s += int(j)
#             m -= s
#     print(m)

# import cv2
# import random
# image = cv2.imread('/Users/wooseoknam/Desktop/Object_Detection/HSV_cylinder.jpg')
# # angle = random.uniform(-45, 45)
# # (h, w) = image.shape[:2]
# # center = (w // 2, h // 2)
# # M = cv2.getRotationMatrix2D(center, 10, 1)
# # image = cv2.warpAffine(image, M, (w, h))
# # print(angle)
# # cv2.imshow('s', image)
# # cv2.waitKey(0)
# crop_ratio = 0.9

# h, w = image.shape[:2]
# crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
# x_min = random.randint(0, w - crop_w)
# y_min = random.randint(0, h - crop_h)
# x_max = x_min + crop_w
# y_max = y_min + crop_h

# cropped_image = image[y_min:y_max, x_min:x_max]

# cv2.imshow('s', cropped_image)
# cv2.waitKey(0)


# import cv2
# import numpy as np

# boxes = [469.5, 707, 925, 584]
# # image
# image = cv2.imread('/Users/wooseoknam/Desktop/lamb.jpeg')
# (h_img, w_img) = image.shape[:2]
# center_img = (w_img // 2, h_img // 2)
# M_img = cv2.getRotationMatrix2D(center_img, -30, 1)
# image = cv2.warpAffine(image, M_img, (w_img, h_img))
# rotation_angle = 30 * np.pi / 180
# rot_matrix = np.array(
#             [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
# print(rot_matrix)
# # box [cx, cy, w, h]
# (h_box, w_box) = boxes[3], boxes[2]
# center_box = (boxes[0], boxes[1])
# M_box = cv2.getRotationMatrix2D(center_box, -30, 1)
# # boxes = cv2.warpAffine(boxes, M_box, (w_box, h_box))
# image = cv2.rectangle(image, (7, 415), (932, 999), (0, 0, 255), 2)

# cv2.imshow('img', image)
# cv2.waitKey(0)



# import cv2
# import numpy as np

# image = cv2.imread('/Users/wooseoknam/Desktop/lamb.jpeg')
# # zoom_factor = 2
# # height, width = image.shape[:2]
# # new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
# # resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# # if zoom_factor > 1:
# #     x_min, y_min = (new_width - width) // 2, (new_height - height) // 2
# #     cropped_image = resized_image[y_min:y_min + height, x_min:x_min + width]
# #     print(resized_image.shape)
# #     print(cropped_image.shape)
# #     cv2.imshow('r', resized_image)
# # else:
# #     x_min, y_min = (width - new_width) // 2, (height - new_height) // 2
# #     padded_image = np.zeros_like(image)
# #     padded_image[y_min:y_min + new_height, x_min:x_min + new_width] = resized_image
# #     cropped_image = padded_image
# #     print(cropped_image.shape)
# # cv2.imshow('z', cropped_image)
# # cv2.waitKey(0)
# _, width, _ = image.shape

# boxes = [469.5, 707, 925, 584]
# boxes[:, 0::2] = width - boxes[:, 2::-2]
# print(boxes)




# import numpy as np

# boxes=np.array([[140.53703951, 75.96141748, 264.72733862, 155.27774932],
#                 [302.71412568, 54.46968814, 388.58173732, 165.06578515],
#                 [236.72949294, 319.26731831, 309.98645493, 373.32594264],
#                 [333.74578115, 234.82155473, 430.14510525, 342.13498547],
#                 [105.77317544, 134.60649828, 288.4534231, 284.23604323]])
# # boxes[:, 0::2] = 640 - boxes[:, 2::-2]
# # print(boxes)
# for i in range(len(boxes)):
#     if boxes[i][1] <= 100:
#         boxes[i] = [boxes[i][0] * 2, boxes[i][1] * 2, boxes[i][2]* 2, boxes[i][3] * 2]

# # for box in boxes:
# #     if box[1] <= 10:
# #         box[1] = 0

# print(boxes)





import xml.etree.ElementTree as ET
from os import listdir
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from files import *
from pascal_voc_writer import Writer
import numpy as np


def read_anntation(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bounding_box_list = []

    file_name = root.find('filename').text
    for obj in root.iter('object'):

        object_label = obj.find("name").text
        for box in obj.findall("bndbox"):
            x_min = int(box.find("xmin").text)
            y_min = int(box.find("ymin").text)
            x_max = int(box.find("xmax").text)
            y_max = int(box.find("ymax").text)

        bounding_box = [object_label, x_min, y_min, x_max, y_max]
        bounding_box_list.append(bounding_box)

    return bounding_box_list, file_name



def read_train_dataset(dir):
    images = []
    annotations = []

    for file in listdir(dir):
        if 'jpg' in file.lower() or 'png' in file.lower():
            images.append(cv2.imread(dir + file, 1))
            annotation_file = file.replace(file.split('.')[-1], 'xml')
            bounding_box_list, file_name = read_anntation(dir + annotation_file)
            annotations.append((bounding_box_list, annotation_file, file_name))

    images = np.array(images)

    return images, annotations


ia.seed(1)


dir = '/Users/wooseoknam/Desktop/CAR_Models/train/'
images, annotations = read_train_dataset(dir)

for idx in range(len(images)):
    image = images[idx]
    boxes = annotations[idx][0]

    ia_bounding_boxes = []
    for box in boxes:
        ia_bounding_boxes.append(ia.BoundingBox(x1=box[1], y1=box[2], x2=box[3], y2=box[4]))

    bbs = ia.BoundingBoxesOnImage(ia_bounding_boxes, shape=image.shape)

    seq = iaa.Sequential([ # crop, noise, hsv, (rotation, flip)(이거 주면 affine 필요 x)
        iaa.Crop(percent=(0, 0.1)),
        iaa.Resize((0.5, 1.0)),
        iaa.SaltAndPepper(0.2),
        iaa.Rotate((-45, 45))
    ], random_order=True)


    seq_det = seq.to_deterministic()

    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    new_image_file = dir + annotations[idx][2][:4] + str(int(annotations[idx][2][4:-4]) + 100) + '.jpg'
    cv2.imwrite(new_image_file, image_aug)

    h, w = np.shape(image_aug)[0:2]
    voc_writer = Writer(new_image_file, w, h)

    for i in range(len(bbs_aug.bounding_boxes)):
        bb_box = bbs_aug.bounding_boxes[i]
        voc_writer.addObject(boxes[i][0], int(bb_box.x1), int(bb_box.y1), int(bb_box.x2), int(bb_box.y2))

    # voc_writer.save(dir + 'after_' + annotations[idx][1])
    voc_writer.save(dir + annotations[idx][2][:4] + str(int(annotations[idx][2][4:-4]) + 100) + '.xml')



# for idx in range(len(images)):
#     image = images[idx]
#     boxes = annotations[idx][0]

#     ia_bounding_boxes = []
#     for box in boxes:
#         ia_bounding_boxes.append(ia.BoundingBox(x1=box[1], y1=box[2], x2=box[3], y2=box[4]))
#     bbs = ia.BoundingBoxesOnImage(ia_bounding_boxes, shape=image.shape)

#     # seq = iaa.Sequential([ # crop, noise, hsv, (rotation, flip)(이거 주면 affine 필요 x)
#     #     iaa.Crop(percent=(0, 0.1)),
#     #     iaa.Sometimes(0.7, iaa.Resize((0.5, 0.75), interpolation=cv2.INTER_AREA)),
#     #     iaa.AdditiveGaussianNoise(scale=(0, 0.01*255), per_channel=0.5), #salt and pepper
#     #     iaa.Multiply([0.7, 0.9], per_channel=0.2),
#     #     iaa.Sometimes(0.1, iaa.Rain()),
#     #     iaa.Sometimes(0.1, iaa.Fog()),
#     #     iaa.Affine(
#     #         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#     #         rotate=(-25, 25),
#     #         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#     #     )
#     # ], random_order=True)

#     seq_det = seq.to_deterministic()

#     image_aug = seq_det.augment_images([image])[0]
#     bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

#     for i in range(len(bbs.bounding_boxes)):
#         before = bbs.bounding_boxes[i]
#         after = bbs_aug.bounding_boxes[i]

#         print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
#             i,
#             before.x1, before.y1, before.x2, before.y2,
#             after.x1, after.y1, after.x2, after.y2)
#         )

#     image_before = bbs.draw_on_image(image)
#     image_after = bbs_aug.draw_on_image(image_aug, color=[0, 0, 255])

#     cv2.imshow('image_before', cv2.resize(image_before, (640, 640)))
#     cv2.imshow('image_after', cv2.resize(image_after, (640, 640)))

#     cv2.waitKey(0)
#     # break