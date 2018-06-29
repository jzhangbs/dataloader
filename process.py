import re
import numpy as np
import sys
import os
import glob
import math
import random
from imageio import imread
from skimage.transform import resize as imresize


def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(br'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = data[::-1, ...]  # cv2.flip(data, 0)
    return data


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)


def gen_driving_path(root_data_folder, root_disp_folder):
    left_folders = []
    right_folders = []
    disp_folders = []
    level1 = ["15mm_focallength", "35mm_focallength"]
    level2 = ["scene_backwards", "scene_forwards"]
    level3 = ["fast", "slow"]
    sides = ["left", "right"]
    for l1 in level1:
        for l2 in level2:
            for l3 in level3:
                left_folders.append(os.path.join(root_data_folder, *[l1, l2, l3, sides[0]]))
                right_folders.append(os.path.join(root_data_folder, *[l1, l2, l3, sides[1]]))
                disp_folders.append(os.path.join(root_disp_folder, *[l1, l2, l3, sides[0]]))

    left_images = []
    right_images = []
    disp_images = []
    for left_folder, right_folder, disp_folder in zip(left_folders, right_folders, disp_folders):
        left_data = sorted(glob.glob(left_folder + "/*.png"))
        right_data = sorted(glob.glob(right_folder + "/*.png"))
        disps_data = sorted(glob.glob(disp_folder + "/*.pfm"))
        left_images = left_images + left_data
        right_images = right_images + right_data
        disp_images = disp_images + disps_data
    return list(zip(left_images, right_images, disp_images))


def gen_monkaa_path(root_data_folder, root_disp_folder):
    left_folders = []
    right_folders = []
    disp_folders = []
    level1 = sorted(os.listdir(root_data_folder))
    sides = ["left", "right"]
    for l1 in level1:
        left_folders.append(os.path.join(root_data_folder, *[l1, sides[0]]))
        right_folders.append(os.path.join(root_data_folder, *[l1, sides[1]]))
        disp_folders.append(os.path.join(root_disp_folder, *[l1, sides[0]]))

    left_images = []
    right_images = []
    disp_images = []
    for left_folder, right_folder, disp_folder in zip(left_folders, right_folders, disp_folders):
        left_data = sorted(glob.glob(left_folder + "/*.png"))
        right_data = sorted(glob.glob(right_folder + "/*.png"))
        disps_data = sorted(glob.glob(disp_folder + "/*.pfm"))
        left_images = left_images + left_data
        right_images = right_images + right_data
        disp_images = disp_images + disps_data
    return list(zip(left_images, right_images, disp_images))


def gen_flyingthings3d_path(root_data_folder, root_disp_folder):
    left_folders = []
    right_folders = []
    disp_folders = []
    level1 = ["TRAIN"]
    level2 = ["A", "B", "C"]
    sides = ["left", "right"]
    for l1 in level1:
        for l2 in level2:
            data_level12 = os.path.join(root_data_folder, *[l1, l2])
            disp_level12 = os.path.join(root_disp_folder, *[l1, l2])
            level3 = sorted(os.listdir(data_level12))
            for l3 in level3:
                left_folders.append(os.path.join(data_level12, *[l3, sides[0]]))
                right_folders.append(os.path.join(data_level12, *[l3, sides[1]]))
                disp_folders.append(os.path.join(disp_level12, *[l3, sides[0]]))

    left_images = []
    right_images = []
    disp_images = []
    for left_folder, right_folder, disp_folder in zip(left_folders, right_folders, disp_folders):
        left_data = sorted(glob.glob(left_folder + "/*.png"))
        right_data = sorted(glob.glob(right_folder + "/*.png"))
        disps_data = sorted(glob.glob(disp_folder + "/*.pfm"))
        left_images = left_images + left_data
        right_images = right_images + right_data
        disp_images = disp_images + disps_data
    return list(zip(left_images, right_images, disp_images))


def gen_sample_list(data_root, dataset_name):
    """output paths in a list: [[img_path1, image_path2, disp_image1], [], ...]"""
    # driving
    driving_data_folder = os.path.join(data_root, 'driving/frames_cleanpass')
    driving_label_folder = os.path.join(data_root, 'driving/disparity')
    # monkaa
    monkaa_data_folder = os.path.join(data_root, 'monkaa/frames_cleanpass')
    monkaa_label_folder = os.path.join(data_root, 'monkaa/disparity')
    # flyingthings3d
    flyingthings3d_data_folder = os.path.join(data_root, 'flyingthings3d/frames_cleanpass')
    flyingthings3d_label_folder = os.path.join(data_root, 'flyingthings3d/disparity')
    # generate the list of all [left, right, disp]
    sample_list = []
    if 'driving' in dataset_name:
        driving_sample_list = gen_driving_path(driving_data_folder, driving_label_folder)
        sample_list = sample_list + driving_sample_list
    if 'monkaa' in dataset_name:
        monkaa_sample_list = gen_monkaa_path(monkaa_data_folder, monkaa_label_folder)
        sample_list = sample_list + monkaa_sample_list
    if 'flyingthings3d' in dataset_name:
        flyingthings3d_sample_list = gen_flyingthings3d_path(flyingthings3d_data_folder, flyingthings3d_label_folder)
        sample_list = sample_list + flyingthings3d_sample_list
    return sample_list


def split_sample_list(sample_list, val_ratio, fraction=1, seed=None):
    random.seed(seed)
    random.shuffle(sample_list)
    num_samples = len(sample_list)
    num_data = int(fraction * num_samples)
    num_validation = int(math.ceil(num_data * val_ratio))
    sample_list = sample_list[0:num_data]
    validation_list = sample_list[0:num_validation]
    training_list = sample_list[num_validation:]
    return training_list, validation_list


def center_image(img):
    """ normalize image """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0,1), keepdims=True)
    mean = np.mean(img, axis=(0,1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)


def read(filename, kwargs):
    ldata, rdata, ddata = filename
    # read images
    left_image = center_image(imread(ldata))
    right_image = center_image(imread(rdata))
    if 'kitti' not in kwargs.dataset_name:
        disp_image = load_pfm(open(ddata, 'rb'))
    else:
        disp_image = imread(ddata, mode='L')
    disp_image = np.expand_dims(disp_image, 2)
    if kwargs.phase == 'val':
        # return stereo pair
        h, w = left_image.shape[0:2]
        # crop_width = w - w % 16
        # crop_height = h - h % 16
        crop_width = kwargs.preproc_args.crop_width
        crop_height = kwargs.preproc_args.crop_height
        return resize([left_image, right_image, disp_image], crop_height, crop_width, kwargs.seed, kwargs.seed * 2)
    elif kwargs.phase == 'train':
        return random_crop([left_image, right_image, disp_image],
                           kwargs.preproc_args.crop_height, kwargs.preproc_args.crop_width)


def random_crop(images, height, width, seed_w=None, seed_h=None, *args, **kwargs):
    left_image, right_image, disp_image = images
    h, w = left_image.shape[0:2]
    random.seed(seed_w)
    start_w = random.randint(0, w - width)
    random.seed(seed_h)
    start_h = random.randint(0, h - height)
    finish_w = start_w + width
    finish_h = start_h + height
    left_image_crop = left_image[start_h:finish_h, start_w:finish_w]
    right_image_crop = right_image[start_h:finish_h, start_w:finish_w]
    disp_image_crop = disp_image[start_h:finish_h, start_w:finish_w]
    return [left_image_crop, right_image_crop, disp_image_crop]


def resize(images, height, width, *args, **kwargs):
    left_image, right_image, disp_image = images

    left_scale = np.max(np.abs(left_image))
    left_image /= left_scale
    left_image_resize = imresize(left_image, (height, width), mode='reflect')
    left_image_resize *= left_scale

    right_scale = np.max(np.abs(right_image))
    right_image /= right_scale
    right_image_resize = imresize(right_image, (height, width), mode='reflect')
    right_image_resize *= right_scale

    return [left_image_resize, right_image_resize, disp_image]
