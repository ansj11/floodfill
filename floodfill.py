
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


def runtime(func, *args):
    def call_func(*args, **kwargs):
        start_time = time()
        print("开始运行: ", start_time)
        out = func(*args, **kwargs)
        print(time() - start_time, "秒")
        return out
    return call_func

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

# DFS搜索
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
        if index[y,x] == num:
            continue # 已经遍历过，则跳过
        index[y, x] = num
        TEST.append([y,x])
        for offset in offsets: # 刚好与递归顺序相反，故TEST不一样
            y1 = min(max(0, y + offset[0]), h - 1)
            x1 = min(max(0, x + offset[1]), w - 1)
            if edge[y1, x1] == 0 or index[y1, x1] > 0:# or [y1,x1] in stack:
                continue  # 非边界或者已经遍历过或者已经加入栈，跳过
            stack.append([y1, x1])
        # plt.imsave('./index/%04d-%04d.jpg'%(y, x), index, vmin=0, vmax=30)
    return index

#
def stack_scanLineFill(x, y, edge, index, num, mode='2'):
    if mode == '2':
        offsets = OFFSETS_2
    else:
        raise NotImplementedError
    h, w = edge.shape[:2]
    stack = [[y,x]]
    while len(stack) > 0:
        y, x = stack.pop()
        if index[y,x] == num:
            continue # 已经遍历过，则跳过
        xl = x
        while xl > 0 and index[y, xl] == 0: # 向左扫描
            index[y, xl] = num
            TEST.append([y,xl])
            for offset in offsets:
                y1 = min(max(0, y + offset[0]), h - 1)
                x1 = min(max(0, xl + offset[1]), w - 1)
                if edge[y1, x1] == 0 or index[y1, x1] > 0:
                    continue  # 非边界或者已经遍历过或者已经加入栈，跳过
                stack.append([y1, x1])
            xl -= 1
            if edge[y, xl] == 0 or index[y, xl] > 0:
                break
        xr = x
        while xr < w-1 and index[y, xl] == 0:  # 向左扫描
            index[y, xr] = num
            TEST.append([y, xr])
            for offset in offsets:
                y1 = min(max(0, y + offset[0]), h - 1)
                x1 = min(max(0, xr + offset[1]), w - 1)
                if edge[y1, x1] == 0 or index[y1, x1] > 0:
                    continue  # 非边界或者已经遍历过或者已经加入栈，跳过
                stack.append([y1, x1])
            xr += 1
            if edge[y, xr] == 0 or index[y, xr] > 0:
                break

        # plt.imsave('./index/%04d-%04d.jpg'%(y, x), index, vmin=0, vmax=30)
    return index

@runtime
def floodfill(edge, index=None, mode='recursive'):
    if index is None:
        index = np.zeros_like(edge)
    h, w = edge.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 25
    videoWriter = cv2.VideoWriter(mode + '.avi', fourcc, fps, (w, h))
    num = 1
    for y in range(h):
        for x in range(w):
            if edge[y,x] != 0 and index[y,x] == 0:
                if mode == 'recursive':
                    index = recursive_floodFill(x, y, edge, index, num)
                elif mode == 'iterative':
                    index = stack_floodFill(x, y, edge, index, num)
                elif mode == 'scanline':
                    index = stack_scanLineFill(x, y, edge, index, num)
                else:
                    raise NotImplementedError
                num += 1
                frame = np.tile(index.reshape(h,w,1), (1,1,3))
                videoWriter.write((frame*9).astype('uint8'))
    videoWriter.release()
    print('联通域个数', num)
    return index

if __name__ == '__main__':
    path = '/Users/anshijie/3dphoto-best2/depth/IMG_9654_edge.jpg'
    edge = cv2.imread(path, 0)
    edge = cv2.resize(edge, (256//1, 256//1), interpolation=cv2.INTER_NEAREST)
    edge[edge <= 127] = 0
    edge[edge > 127] = 1
    plt.imsave('./edge.jpg', edge)
    # index = floodfill(edge, mode='recursive')
    # index = floodfill(edge, mode='iterative')
    index = floodfill(edge, mode='scanline')
    plt.imsave('./index.jpg', index)
    # embed()
    # recursive：0.321s;12156
    # stack：0.285s;12156
    # scanLine:0.252s；13541
