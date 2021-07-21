
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from time import time

TEST = []
OFFSETS_2 = [[-1, 0], [1, 0]]
OFFSETS_4 = [[0, -1], [-1, 0], [1, 0], [0, 1]]
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1,  0], [1,  0],
             [-1,  1], [0,  1], [1,  1]]

# 迭代方式会超过最大递归次数，64直接报错，因此不可行
def recursive_floodFill(x, y, edge, index, num, mode='4'):
    if mode == '4':
        offsets = OFFSETS_4
    elif mode == '8':
        offsets = OFFSETS_8
    else:
        raise NotImplementedError
    if edge[y,x] == 0 or index[y,x] > 0:
        return index
    h, w = edge.shape[:2]
    index[y,x] = num
    TEST.append([y, x])
    for offset in offsets:
        y1 = min(max(0, y+offset[0]), h-1)
        x1 = min(max(0, x+offset[1]), w-1)
        if edge[y1,x1] == 0 or index[y1,x1] > 0:
            continue # 非边界或者已经遍历过，跳过
        index = recursive_floodFill(x1, y1, edge, index, num)

    return index

def stack_floodFill(x, y, edge, index, num, mode='4'):
    if mode == '4':
        offsets = OFFSETS_4
    elif mode == '8':
        offsets = OFFSETS_8
    else:
        raise NotImplementedError
    h, w = edge.shape[:2]
    stack = [[y,x]]
    while len(stack) > 0:
        y, x = stack.pop()
        index[y, x] = num
        TEST.append([y,x])
        for offset in offsets:
            y1 = min(max(0, y + offset[0]), h - 1)
            x1 = min(max(0, x + offset[1]), w - 1)
            if edge[y1, x1] == 0 or index[y1, x1] > 0 or [y1,x1] not in stack:
                continue  # 非边界或者已经遍历过或者已经加入栈，跳过
            stack.append([y1, x1]) # 避免重复加入栈会加速

        # plt.imsave('./index/%04d-%04d.jpg'%(y, x), index, vmin=0, vmax=30)
    return index

也比较慢，64很慢
def stack_scanLineFill(x, y, edge, index, num, mode='2'):
    if mode == '2':
        offsets = OFFSETS_2
    else:
        raise NotImplementedError
    h, w = edge.shape[:2]
    stack = [[y,x]]
    while len(stack) > 0:
        y, x = stack.pop()
        xl = x
        while xl > 0: # 向左扫描
            index[y, xl] = num
            TEST.append([y,xl])
            for offset in offsets:
                y1 = min(max(0, y + offset[0]), h - 1)
                x1 = min(max(0, xl + offset[1]), w - 1)
                if edge[y1, x1] == 0 or index[y1, x1] > 0 or [y1,x1] not in stack:
                    continue  # 非边界或者已经遍历过或者已经加入栈，跳过
                stack.append([y1, x1])
            xl -= 1
            if edge[y, xl] == 0 or index[y, xl] > 0:
                break
        xr = x
        while xr < w-1:  # 向左扫描
            index[y, xr] = num
            TEST.append([y, xr])
            for offset in offsets:
                y1 = min(max(0, y + offset[0]), h - 1)
                x1 = min(max(0, xr + offset[1]), w - 1)
                if edge[y1, x1] == 0 or index[y1, x1] > 0 or [y1, x1] not in stack:
                    continue  # 非边界或者已经遍历过或者已经加入栈，跳过
                stack.append([y1, x1])
            xr += 1
            if edge[y, xr] == 0 or index[y, xr] > 0:
                break

        # plt.imsave('./index/%04d-%04d.jpg'%(y, x), index, vmin=0, vmax=30)
    return index

def floodfill(edge, index=None):
    if index is None:
        index = np.zeros_like(edge)
    h, w = edge.shape[:2]
    num = 1
    for y in range(h):
        for x in range(w):
            if edge[y,x] != 0 and index[y,x] == 0:
                # index = recursive_floodFill(x, y, edge, index, num)
                index = stack_floodFill(x, y, edge, index, num)
                # index = stack_scanLineFill(x, y, edge, index, num)
                num += 1
    return index

if __name__ == '__main__':
    path = '/Users/anshijie/3dphoto-best2/depth/IMG_9654_edge.jpg'
    edge = cv2.imread(path, 0)
    edge = cv2.resize(edge, (256//1, 256//1), interpolation=cv2.INTER_NEAREST)
    edge[edge <= 127] = 0
    edge[edge > 127] = 1
    plt.imsave('./edge.jpg', edge)
    s = time()
    index = floodfill(edge)
    print(time()-s)
    plt.imsave('./index.jpg', index)
    embed()
    # recursive：0.321s;12156
    # stack：0.285s;12156
    # scanLine:
