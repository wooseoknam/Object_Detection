# import xml.etree.ElementTree as ET

# tree = ET.parse('/Users/wooseoknam/Desktop/Object_Detection/YOLOX/datasets/VOCdevkit/VOC2007/Annotations/car_1.xml')
# root = tree.getroot()

# bounding_box_list = []

# file_name = root.find('filename').text
# for obj in root.iter('object'):

#     object_label = obj.find("name").text
#     for box in obj.findall("bndbox"):
#         x_min = int(box.find("xmin").text)
#         y_min = int(box.find("ymin").text)
#         x_max = int(box.find("xmax").text)
#         y_max = int(box.find("ymax").text)

#     bounding_box = [object_label, x_min, y_min, x_max, y_max]
#     bounding_box_list.append(bounding_box)

# print(bounding_box_list)

# import numpy as np
# import cv2

# img = cv2.imread('/Users/wooseoknam/Desktop/Object_Detection/crawled_img/car_1.jpg')

# hsv_augs = np.random.uniform(-1, 1, 3) * [5, 30, 30]  # random gains
# hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
# hsv_augs = hsv_augs.astype(np.int16)
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

# img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
# img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
# img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

# cv2.imshow('a', cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img))  # no return needed

# # cv2.imshow('a', img_hsv)
# cv2.waitKey(0)






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
#     if '-' in equation[0]:
#         m = int(equation[0])
#     elif '-' not in equation[0]:
#         equation[0] = equation[0].split('+')
#         _lst = []
#         for i in equation[0]:
#             i = int(i)
#             _lst.append(i)
#         m = sum(_lst)
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



# N = int(input())
# A = list(map(int, input().split()))
# B = list(map(int, input().split()))

# A_sorted = A.sort(reverse=True)
# B_sorted = B.sort()

# S = 0
# for i in range(N):
#     S += (A[i] * B[i])
# print(S)



# N = int(input())
# lst = []
# i = 1

# while (sum(lst) + i) < N:
#     lst.append(i)
#     i += 1

# lst = lst[:-1]
# last = N - (sum(lst))
# lst.append(last)
# print(len(lst))

# s = int(input())
# cnt = 0

# while s > cnt:
#     cnt += 1
#     s -= cnt

# print(cnt)



# coin = [500, 100, 50, 10, 5, 1]
# ans = []
# price = int(input())
# cng = 1000 - price

# for i in coin:
#     ans.append(cng // i)
#     cng -= ((cng // i) * i)

# print(sum(ans))



# lst = [300, 60, 10]

# T = int(input())
# ans = []

# if str(T)[-1] != '0':
#     print(-1)
# else:
#     for i in lst:
#         ans.append(T // i)
#         T -= ((T // i) * i)
#     print(*ans)


# N = int(input())
# lst = []
# for i in range(N):
#     lst.append(int(input()))
# print(min(lst) * N)



# def is_30(n):
#     if (sum(map(int, n)) % 3 == 0) and (N[-1] == '0'):
#         return True
#     else:
#         return False
    
# N = input()

# if is_30(N) == False:
#     print(-1)
# else:
#     lst = list(str(N))
#     lst.sort(reverse=True)
#     print(int(''.join(lst)))


# N = input()
# lst = list(N)
# lst.sort(reverse=True)
# sorted_N = ''.join(lst)

# if (sum(map(int, sorted_N)) % 3 == 0) and (str(sorted_N)[-1] == '0'):
#     print(int(sorted_N))
# else:
#     print(-1)



# N, M = map(int, input().split())
# x6_lst = []
# x1_lst = []
# S = 0

# for i in range(M):
#     x6, x1 = map(int, input().split())
#     x6_lst.append(x6)
#     x1_lst.append(x1)

# S += min(x6_lst) * (N // 6)
# N -= (N // 6) * 6
# S += min(x1_lst) * N

# print(S)


# str_ = input()
# print(1) if str_[::-1] == str_ else print(0)