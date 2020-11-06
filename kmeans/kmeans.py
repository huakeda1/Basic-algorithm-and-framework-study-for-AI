#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import random
import matplotlib.pyplot as plt


def generate_data(data_number):
    return [random.randint(0, 100) for _ in range(data_number)], [random.randint(0, 100) for _ in range(data_number)]


def distant(point_a, point_b):
    return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def random_color(number):
    color_arr = [str(i) for i in range(10)] + [chr(x + ord('A')) for x in range(6)]
    color_map = []
    for i in range(number):
        color_single = '#'
        for j in range(6):
            index = random.randint(0, 15)
            color_single += color_arr[index]
        color_map.append(color_single)
    return color_map


def get_cluster_centers(raw_data, cluster_number, distant_func):
    previous_cluster_centers = random.sample(raw_data, cluster_number)
    color_map = random_color(cluster_number)
    while True:
        groups = [[] for _ in range(cluster_number)]
        for data in raw_data:
            distances = [distant_func(data, center) for center in previous_cluster_centers]
            index = np.argmin(distances)
            groups[index].append(data)
        new_cluster_centers = []
        plt.cla()
        for i, group in enumerate(groups):
            group_x = [x for x, y in group]
            group_y = [y for x, y in group]
            new_cluster_centers.append([np.mean(group_x), np.mean(group_y)])
            plt.scatter(group_x, group_y, s=25, c=color_map[i])
            plt.scatter([np.mean(group_x)], [np.mean(group_y)], s=400, c=color_map[i], alpha=0.3)
        plt.pause(0.5)
        if sum([distant(point_a, point_b) for point_a, point_b in
                zip(previous_cluster_centers, new_cluster_centers)]) < 0.00001:
            previous_cluster_centers = new_cluster_centers
            break
        else:
            previous_cluster_centers = new_cluster_centers
    plt.show()
    return previous_cluster_centers


data_number = 1000
random_x, random_y = generate_data(data_number)

random_data = [[x, y] for x, y in zip(random_x, random_y)]

cluster_number = 5

plt.ion()

cluster_centers = get_cluster_centers(random_data, cluster_number, distant_func=distant)
print(cluster_centers)
print(len(cluster_centers))
