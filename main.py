from PIL import Image
import cv2
import numpy as np
from datasets import load_dataset
import random
from tqdm import tqdm


ds = load_dataset("Chris1/cityscapes")


def RGB2gray(img: np.array):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def RGB2Lab(img: np.array):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def RGB2YCC(img: np.array):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)


def RGB2HSL(img: np.array):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)


def get_same_color_pixels(img, color):
    return (img == color * np.ones(img.shape)).astype(np.uint8) * 255


def is_inside(coord, shape):
    return 0 <= coord[0] < shape[0] and 0 <= coord[1] < shape[1]


def tuple_sum(t1, t2):
    if len(t1) != len(t2):
        raise ValueError
    return tuple(t1[i] + t2[i] for i in range(len(t1)))


def intersection_over_union(pred: np.array, true: np.array):
    res = {}
    for c in tqdm(np.unique(pred), desc="calculating IoU"):
        res |= {c: np.vectorize(lambda x, y: x == c and y == c)(pred, true).sum() / (np.vectorize(lambda x, y: x == c or y == c)(pred, true)).sum()}
    wmean = sum(res[c] * (true == c).sum() for c in res)
    res |= {'mean': sum(res.values()) / len(res)}
    res |= {'weighted_mean': wmean / (1 << 21)}
    return res


def process_image(img: np.array, class_map: np.array, criteria):
    seeds = []
    label_dict = {}
    for l in tqdm(np.unique(class_map), desc="finding centroids"):
        seeds += (curr := find_centroids(get_same_color_pixels(class_map, l)))
        if l:
            label_dict |= {tuple(seed.tolist()): l for seed in curr}

    return regional_growth(img, label_dict, criteria)


def find_centroids(cmap: np.array):
    *_, centroids = cv2.connectedComponentsWithStats(cmap)

    res = []
    for c in centroids.astype(int):
        c = c[::-1]
        if is_inside(c, cmap.shape) and cmap[*c] != 0:
            res.append(c)
    return res


def regional_growth(img: np.array, labels: dict, grow_criteria: callable):
    directions = tuple((x, y) for x in range(-2, 3) for y in range(-2, 3) if x and y)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for current_point, label in tqdm(labels.items(), desc=f"class growing"):
        current_class = [current_point]

        for x in range(-4, 5):
            for y in range(-4, 5):
                if is_inside(p := tuple_sum(current_point, (x, y)), img.shape):
                    current_class.append(p)
                    mask[*p] = label
        while current_class:
            current_point = current_class.pop()
            for direction in random.sample(directions, 6):
                new_point = tuple_sum(current_point, direction)
                if is_inside(new_point, img.shape) and not mask[*new_point] and grow_criteria(img[*new_point],
                                                                                              img[*current_point]):
                    current_class.append(new_point)
                    mask[*new_point] = label

    return mask


def show_image(img: np.array):
    cv2.imshow('window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


criteria = lambda x, y: max(x[1], y[1]) - min(x[1], y[1]) <= 3
img_res = process_image(img := RGB2HSL(imgRGB := np.array(ds['validation'].select([149])['image'][0])),
                        img_map := RGB2gray(np.array(ds['validation'].select([149])['semantic_segmentation'])[0]), criteria)

show_image(imgRGB[:, :, ::-1])
show_image(img_map)
show_image(img_res)
show_image(imgRGB[:, :, ::-1] * np.array([bl := (img_res == 0), bl, bl]).transpose(1,2,0))

c_img = img_map.copy()

seeds = []
for l in np.unique(img_map):
    seeds += find_centroids(get_same_color_pixels(img_map, l))

for c in seeds:
    print(c)
    for x in range(-2, 3):
        for y in range(-2, 3):
            c_img[c[0] + x, c[1] + y] = 255
show_image(c_img)

cv2.imwrite('source.jpg', imgRGB[:, :, ::-1])
cv2.imwrite('mask.jpg', img_res)
cv2.imwrite('compare.jpg', imgRGB[:, :, ::-1] * np.array([bl := (img_res == 0), bl, bl]).transpose(1, 2, 0))
cv2.imwrite('centroids.jpg', c_img)
