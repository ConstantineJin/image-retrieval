import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# 读入文件
def read_image(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        # 读入图像并将其转化为灰度图像
        image = cv2.imread(os.path.join(folder_path, filename), 0)
        if image is not None:
            images.append(image)
    return images


def extract_sift_features(img):
    # 提取SIFT特征
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = sift.detectAndCompute(img, None)
    return descriptors


def build_vocabulary(descriptors, k):
    # 构建视觉单词
    kmeans = KMeans(n_clusters=k, random_state=0).fit(descriptors)
    return kmeans.cluster_centers_


def get_histogram(descriptor, vocabulary):
    # 计算频率直方图
    distances = cdist(descriptor, vocabulary, metric='euclidean')
    min_index = np.argmin(distances, axis=1)
    histogram, _ = np.histogram(min_index, bins=len(vocabulary), range=(0, len(vocabulary)))
    return histogram


def search_similar_images(query_img, dataset, k):
    # 提取查询图片和数据集中的所有图片的SIFT特征
    query_descriptors = extract_sift_features(query_img)
    dataset_descriptors = [extract_sift_features(img) for img in dataset]

    # 构建词汇表
    descriptors = np.vstack(dataset_descriptors)
    vocabulary = build_vocabulary(descriptors, k)

    # 计算频率直方图
    query_histogram = get_histogram(query_descriptors, vocabulary)
    dataset_histograms = [get_histogram(descriptor, vocabulary) for descriptor in dataset_descriptors]

    # 计算相似度
    similarities = [np.dot(query_histogram, histogram) / (np.linalg.norm(query_histogram) * np.linalg.norm(histogram))
                    for histogram in dataset_histograms]

    # 排序结果
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_images = [dataset[i] for i in sorted_indices]

    return sorted_images


# 读入文件
dataset = read_image('./oxbuild_images')
"""
# 完整数据集处理时间较长，可用较小的数据集先行验证
dataset = [cv2.imread('./oxbuild_images/all_souls_000066.jpg'), cv2.imread('./oxbuild_images/all_souls_000051.jpg'),
           cv2.imread('./oxbuild_images/all_souls_000093.jpg')]
"""

# 需要查找的图片
query_img = cv2.imread('./oxbuild_images/all_souls_000065.jpg')

# 词汇表数量
k = 50

similar_images = search_similar_images(query_img, dataset, k)

fig, axs = plt.subplots(1, len(similar_images) + 1)
axs[0].imshow(query_img[:, :, ::-1])
axs[0].axis('off')
for i in range(len(similar_images)):
    axs[i + 1].imshow(similar_images[i][:, :, ::-1])
    axs[i + 1].axis('off')
plt.show()
