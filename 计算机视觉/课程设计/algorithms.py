import cv2
import numpy as np
from typing import List, Union, Dict
from matplotlib import pyplot as plt
from dataclasses import dataclass


def blur(image_arr, **kwargs):
    if kwargs['algorithm'] == 'gaussian':
        return cv2.GaussianBlur(image_arr, kwargs['ksize'], kwargs['sigmaX'], kwargs['sigmaY'])
    if kwargs['algorithm'] == 'bilateral':
        return cv2.bilateralFilter(image_arr, kwargs['d'], kwargs['sigmaColor'], kwargs['sigmaSpace'])


canny = cv2.Canny


def watershed(img_blur, img_canny, ):
    def get_marker(img_canny):
        contours = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0]
        # Create the marker image for the watershed algorithm
        markers = np.zeros(img_canny.shape, dtype=np.int32)
        # Draw the foreground markers
        for i in range(len(contours)):
            cv2.drawContours(markers, contours, i, (i + 1), -1)
        return markers

    marker = get_marker(img_canny)
    return cv2.watershed(img_blur, marker)


class ConnectSet:
    idx: int  # the index of a set
    points: np.ndarray
    point_indices: List[np.ndarray]  # the index of points
    head: 'ConnectSet'

    def __init__(self, idx, points, point_indices):
        self.idx = idx
        self.points = points
        self.point_indices = point_indices
        self.head = self

    def find_head(self):
        # the head of a ConnectSet should point to itself
        head = self.head
        while head.idx != head.head.idx:
            head = head.head
        return head

    def set_head(self, head: 'ConnectSet'):
        # 1. flatten a connect set, 2. used in merge
        node = self
        while node.head.idx != node.idx:
            prev = node
            node = node.head
            prev.head = head
        node.head = head

    def merge(self, a: 'ConnectSet'):
        head1 = self.find_head()
        head2 = a.find_head()
        head1.points = np.vstack([head1.points, head2.points])
        head1.point_indices[0] = np.hstack([head1.point_indices[0], head2.point_indices[0]])
        head1.point_indices[1] = np.hstack([head1.point_indices[1], head2.point_indices[1]])
        self.set_head(head1)  # flatten connect set
        a.set_head(head1)  # merge two regions


class RegionMerge:
    region_num: int
    regions: List[ConnectSet]
    heap: List

    def __init__(self):
        return

    def fit(self, img_blur, marker, threshold, min_regions):
        self._scan(img_blur, marker)
        size = len(list(set([x.find_head().idx for x in self.regions])))
        while self.heap[0][2] < threshold and size >= min_regions:
            print(self.heap[0][2])
            self._merge()
            size = len(list(set([x.find_head().idx for x in self.regions])))
        indices = list(set([x.find_head().idx for x in self.regions]))
        contours = [self.regions[x].point_indices for x in indices]
        new_marker = np.zeros_like(marker, dtype=np.int)
        for idx, point_indices in enumerate(contours):
            new_marker[tuple(point_indices)] = idx
        return new_marker

    def _scan(self, img_blur, marker):
        """inner functions to create regions"""
        self.region_num = np.max(marker + 1)
        # initialize the regions and heap
        self.regions: List[ConnectSet] = []
        self.heap = []
        for idx in range(self.region_num):
            points = img_blur[marker == idx]
            self.regions.append(ConnectSet(idx, points, list(np.where(marker == idx))))

        neighbours: Dict[int, list] = {}  # idx1 connect to idx2 and idx1 < idx2
        m, n = marker.shape
        for i in range(m):
            for j in range(n):
                if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                    continue
                if marker[i, j] == -1:
                    x_idx = [i - 1, i - 1, i - 1, i, i, i + 1, i + 1, i + 1,]
                    y_idx = [j - 1, j, j + 1, j - 1, j + 1, j-1, j,  j + 1]
                    connected_points = [x for x in marker[tuple([x_idx, y_idx])] if x != -1]
                    connected_points.sort()
                    for idx, k in enumerate(connected_points):
                        try:
                            neighbours[k]
                        except KeyError:
                            neighbours[k] = connected_points[idx + 1:]
                        else:
                            neighbours[k] += connected_points[idx + 1:]
                    self.regions[connected_points[0]].points = np.vstack([self.regions[connected_points[0]].points,
                                                                          img_blur[i, j]])
                    np.append(self.regions[connected_points[0]].point_indices[0], i)
                    np.append(self.regions[connected_points[0]].point_indices[1], j)

        for idx, neighbours in neighbours.items():
            for neighbour in neighbours:
                if idx == neighbour:
                    continue
                disimilarity = self._compute_disimiarity(self.regions[idx].points, self.regions[neighbour].points)
                self.heap.append([idx, neighbour, disimilarity])
        self.heap.sort(key=lambda x: x[2])

    def _merge(self):
        """inner function to merge two regions"""
        self.region_num -= 1
        region_1_idx, region_2_idx, _ = self.heap[0]
        del self.heap[0]
        region_1 = self.regions[region_1_idx]
        region_2 = self.regions[region_2_idx]
        region_1.merge(region_2)
        # update the heap
        for idx, node in enumerate(self.heap):
            region1_idx, region2_idx, disimiarity = node
            region1 = self.regions[region1_idx].find_head()
            region2 = self.regions[region2_idx].find_head()
            if region1_idx == region1.idx and region2_idx == region2.idx:
                continue
            else:
                disimiarity = self._compute_disimiarity(region1.points, region2.points)
                self.heap[idx] = [region1.idx, region2.idx, disimiarity]
        self.heap.sort(key=lambda x: x[2])
        # remove redundant elements in the list
        cmp = self.heap[-1]
        for idx in range(len(self.heap) - 2, -1, -1):
            if (self.heap[idx][0] == cmp[0] and self.heap[idx][1] == cmp[1]) or (
                    self.heap[idx][0] == cmp[1] or self.heap[idx][1] == cmp[0]):
                del self.heap[idx]
            else:
                cmp = self.heap[idx]

    @staticmethod
    def _compute_disimiarity(points_1, points_2):
        return points_1.size * points_2.size / (points_1.size + points_2.size) * \
               np.sum((np.mean(points_1, axis=0) - np.mean(points_2, axis=0)) ** 2)


def color_generator():
    while True:
        yield np.random.randint(low=0., high=255., size=(3,))


def draw_marker(marker):
    generator = color_generator()
    img_arr = np.zeros(marker.shape + (3,), dtype=np.float64)
    for i in range(np.max(marker) + 1):
        img_arr[marker == i] = next(generator)
    return img_arr


if __name__ == '__main__':
    filepath = 'D:/data/Image1.jpg'
    img_arr = cv2.imread(filepath)
    img_blur = blur(img_arr, algorithm='gaussian', ksize=(5, 5), sigmaX=0, sigmaY=0)
    img_canny = canny(img_blur, 80, 120)
    marker = watershed(img_blur, img_canny)
    img_watershed = draw_marker(marker)

    region_merge = RegionMerge()
    marker_merged = region_merge.fit(img_blur, marker, 16000000, 1)
    img_merged = draw_marker(marker_merged)

    cv2.imshow('watershed', img_watershed / 255)
    cv2.imshow('merged', img_merged/255)
    cv2.waitKey(0)
    plt.show()
