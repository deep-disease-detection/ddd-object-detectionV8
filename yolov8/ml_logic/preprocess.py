# import std libraries
import os
import csv
import sys
from itertools import product

# import data processing libraries
import numpy as np

# import images processing libraries
import cv2

# import sklearn classes and function
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# import keras classes and functions
from tensorflow import keras
from keras.utils import to_categorical
import tensorflow as tf
from shutil import copyfile

# pour la fonction convert base64
from imageio import imread
import base64
import io

# import classes and functions from ddd
from yolov8.params import *
from ddd.params import *
from ddd.ml_logic.preprocess import get_pic_mesure, resize_image
from ddd.utils import *
import math


def get_dic_center_yolo(split_set: str = "train"):
    """
    Create a dictionary with all coordinates center for each virus and each file
    """
    # dictionary initiation
    train_data_dic = {}
    list_virus = [
        v for v in os.listdir(os.path.join(RAW_DATA_PATH, split_set)) if v[0] != "."
    ]

    # filling the dictinary virus by virus
    for virus in list_virus:
        train_data_dic[virus] = {}
        path_position_file = os.path.join(
            RAW_DATA_PATH, split_set, virus, "particle_positions"
        )

        # getting all anotation files
        list_position_file = [f for f in os.listdir(path_position_file) if f[0] != "."]
        # looping over each annotation file
        for file in list_position_file:
            with open(
                os.path.join(
                    RAW_DATA_PATH, split_set, virus, "particle_positions", file
                ),
                "r",
            ) as f:
                lines = f.readlines()
                particles = []
                particle = []
                for i in range(3, len(lines)):
                    if lines[i] != "particleposition\n":
                        coordinate = tuple(
                            float(c) for c in lines[i].strip("\n").split(";")
                        )
                        particle.append(coordinate)

                        # handle last particle of file
                        if i == len(lines) - 1 and len(particle) == 2:
                            particles.append([average_coord(particle[0], particle[1])])
                        elif i == len(lines) - 1:
                            particles.append(particle)
                    else:
                        # compute the center between 2 center point of one single particle
                        if len(particle) == 2:
                            particles.append([average_coord(particle[0], particle[1])])
                            particle = []
                        else:
                            particles.append(particle)
                            particle = []
                file_name = file.replace("_particlepositions.txt", "")
                train_data_dic[virus][file_name] = particles

    return train_data_dic


def resize_particles_yolo(particles: list, file: str, pic_dict: dict):
    return [
        [
            (c[0] * pic_dict[file]["Yscale"], c[1] * pic_dict[file]["Xscale"])
            for c in particle
        ]
        for particle in particles
    ]


def check_box_in_pic(x, y, w, h):
    return (
        x - w / 2 >= 0
        and x + w / 2 <= YOLO_IMAGE_SIZE
        and y - h / 2 >= 0
        and y + h / 2 <= YOLO_IMAGE_SIZE
    )


def preprocess_yolo(split_set: str = "train"):

    particle_dict = get_dic_center_yolo(split_set)
    pic_dict = get_pic_mesure(split_set)

    for virus in VIRUS_METADATA.keys():

        print(f"Preparing {virus} for YOLOv7 🦠")

        image_file_count = len(pic_dict.get(virus))
        for i, image_file_name in enumerate(pic_dict.get(virus)):

            print(f"--------- Processing image {i+1}/{image_file_count} 🎞️")

            image_path = os.path.join(
                RAW_DATA_PATH, split_set, virus, f"{image_file_name}.tif"
            )

            img = cv2.imread(image_path, -1).astype(np.float32)

            # handle images that have 3 channels
            if len(img.shape) == 2:
                img = np.expand_dims(cv2.imread(image_path, -1).astype(np.float32), 2)
            elif len(img.shape) > 2:
                img = np.expand_dims(img[:, :, 1], 2)

            # get particles
            particles = particle_dict.get(virus).get(image_file_name)

            # resize image and particles
            img = resize_image(img, image_file_name, pic_dict.get(virus))
            r_particles = resize_particles_yolo(
                particles, image_file_name, pic_dict.get(virus)
            )

            # crop images to 1000x1000 pixels
            x_coordinates = []
            y_coordinates = []
            for particle in r_particles:
                x_coordinates += [p[0] for p in particle]
                y_coordinates += [p[1] for p in particle]

            # compute mean of all particles
            central_particle = (
                int(np.mean(x_coordinates)),
                int(np.mean(y_coordinates)),
            )

            # compute the margin we need to pad if necessary
            top = central_particle[1] + YOLO_IMAGE_SIZE / 2
            bottom = central_particle[1] - YOLO_IMAGE_SIZE / 2
            right = central_particle[0] + YOLO_IMAGE_SIZE / 2
            left = central_particle[0] - YOLO_IMAGE_SIZE / 2

            border_top = int(max(top - img.shape[0], 0))
            border_bottom = int(abs(min(bottom, 0)))
            border_right = int(max(right - img.shape[1], 0))
            border_left = int(abs(min(left, 0)))

            # pad if necessary
            crop_img = cv2.copyMakeBorder(
                img,
                border_bottom,
                border_top,
                border_left,
                border_right,
                borderType=cv2.BORDER_CONSTANT,
            )

            # crop
            crop_img = crop_img[
                int(bottom + border_bottom) : int(top + border_bottom),
                int(left + border_left) : int(right + border_left),
            ]

            # adapt particle positions to crop
            final_particles = [
                [(p[0] - left, p[1] - bottom) for p in particle]
                for particle in r_particles
            ]

            # make the bounding boxes
            image_labels = []

            for f_particle in final_particles:
                # pick a color for the particle

                # if there is only one coordinate, we make a box using virus metadata
                if len(f_particle) == 1:

                    # if only 1 coordinate set and elongated, simply skip
                    if VIRUS_METADATA.get(virus).get("elongated"):
                        continue

                    yolo_x = f_particle[0][0]
                    yolo_y = f_particle[0][1]
                    yolo_w = VIRUS_METADATA.get(virus)["diameter"]
                    yolo_h = VIRUS_METADATA.get(virus)["diameter"]

                    if not check_box_in_pic(yolo_x, yolo_y, yolo_w, yolo_h):
                        continue

                    # Normalize x,y, w and h
                    yolo_x = yolo_x / YOLO_IMAGE_SIZE
                    yolo_y = yolo_y / YOLO_IMAGE_SIZE
                    yolo_h = yolo_h / YOLO_IMAGE_SIZE
                    yolo_w = yolo_w / YOLO_IMAGE_SIZE

                    image_labels.append(
                        f'{VIRUS_METADATA.get(virus).get("id")} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n'
                    )

                else:
                    x_coordinates = [p[0] for p in f_particle]
                    y_coordinates = [p[1] for p in f_particle]

                    margin = VIRUS_METADATA.get(virus)["diameter"]
                    min_x = min(x_coordinates) - margin
                    max_x = max(x_coordinates) + margin
                    min_y = min(y_coordinates) - margin
                    max_y = max(y_coordinates) + margin
                    xy = (min_x, min_y)
                    w = max_x - min_x
                    h = max_y - min_y

                    yolo_x = xy[0] + w / 2
                    yolo_y = xy[1] + h / 2
                    yolo_w = w
                    yolo_h = h

                    if not check_box_in_pic(yolo_x, yolo_y, yolo_w, yolo_h):
                        continue

                    # Normalize x,y, w and h
                    yolo_x = yolo_x / YOLO_IMAGE_SIZE
                    yolo_y = yolo_y / YOLO_IMAGE_SIZE
                    yolo_h = yolo_h / YOLO_IMAGE_SIZE
                    yolo_w = yolo_w / YOLO_IMAGE_SIZE

                    image_labels.append(
                        f'{VIRUS_METADATA.get(virus).get("id")} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n'
                    )

            # if at least one particle makes it, save the image and the file
            if len(image_labels) > 0:
                crop_img = (
                    cv2.normalize(crop_img, None, 1.0, 0.0, cv2.NORM_MINMAX) * 255
                ).astype(np.uint8)

                # save the cropped image
                if split_set == "validation":
                    split_set_save = "valid"
                else:
                    split_set_save = split_set

                saved_image_path = os.path.join(
                    YOLO_DATA_PATH,
                    "images",
                    split_set_save,
                    f"{virus}_{image_file_name}.jpg",
                )
                cv2.imwrite(saved_image_path, crop_img)

                # save the label file
                with open(
                    os.path.join(
                        YOLO_DATA_PATH,
                        "labels",
                        split_set_save,
                        f"{virus}_{image_file_name}.txt",
                    ),
                    "w",
                ) as f:
                    f.writelines(image_labels)


def new_coordinates_flip(x1, y1, flip_param):
    if flip_param == 0:
        x2 = x1
        y2 = 2 * 0.5 - y1
    elif flip_param == 1:
        x2 = 2 * 0.5 - x1
        y2 = y1
    elif flip_param == -1:
        x2 = 2 * 0.5 - x1
        y2 = 2 * 0.5 - y1
    return x2, y2


def new_coordinates_rotation(x1, y1, degree, width, height):
    x2 = ((x1 - 0.5) * math.cos(degree)) - ((y1 - 0.5) * math.sin(degree)) + 0.5
    y2 = ((x1 - 0.5) * math.sin(degree)) - ((y1 - 0.5) * math.cos(degree)) + 0.5
    new_width = height
    new_height = width
    return x2, y2, new_width, new_height


def get_label_file(file_name):
    file_name_in_txt = file_name.replace("jpg", "txt")
    label_file = os.path.join(YOLO_DATA_PATH, "labels", "train", file_name_in_txt)
    return label_file


def image_augmented(image_name, degree, flip_param, nbre_passage):
    if degree == math.pi / 2:
        degree = cv2.ROTATE_90_CLOCKWISE
    elif degree == math.pi:
        degree = cv2.ROTATE_180
    else:
        degree = cv2.ROTATE_90_COUNTERCLOCKWISE
    image = cv2.imread(os.path.join(YOLO_DATA_PATH, "images", "train", image_name))
    new_image = cv2.rotate(image, degree)
    new_image = cv2.flip(new_image, flip_param)
    cv2.imwrite(
        os.path.join(
            YOLO_DATA_PATH, "images", "augmented", f"AUG{nbre_passage}_{image_name}"
        ),
        new_image,
    )


def dictionary_initialization(yolo_image_train_path):
    """compute images to create for yolo training"""

    dic = {}
    nb_to_create = {}

    for virus in VIRUSES:
        dic[virus] = 0

    for file in os.listdir(yolo_image_train_path):
        ind = file.find("_")
        virus_name = file[:ind]
        dic[virus_name] = dic[virus_name] + 1

    for virus in VIRUSES:
        nb_to_create[virus] = 100 - dic[virus]

    # removing unuseful virus
    del nb_to_create["Influenza"]
    del nb_to_create["Lassa"]
    del nb_to_create["Nipah virus"]

    return nb_to_create


def new_label_documentation(file_name, file_path, rotate, flip, nbre_passage):
    particle = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            label = line[0]
            x = float(line[1])
            y = float(line[2])
            width = line[3]
            height = line[4]
            # rotation
            new_x, new_y, new_width, new_height = new_coordinates_rotation(
                x, y, rotate, width, height
            )
            # flipping
            new_x, new_y = new_coordinates_flip(new_x, new_y, flip)
            particle.append(f"{label} {new_x} {new_y} {new_width} {new_height} \n")

    # save the label file
    with open(
        os.path.join(
            YOLO_DATA_PATH,
            "labels",
            "augmented",
            f'AUG{nbre_passage}_{file_name.replace("jpg","txt")}',
        ),
        "w",
    ) as f:
        f.writelines(particle)


# augmentation de YOLO
def augmentation_of_yolo():

    # creation of the augmented directory if it does not exist
    try:
        os.listdir(os.path.join(YOLO_DATA_PATH, "images", "augmented"))
    except FileNotFoundError:
        print("There is no augmented train directory, creating it")
        os.makedirs(os.path.join(YOLO_DATA_PATH, "images", "augmented"))

    try:
        os.listdir(os.path.join(YOLO_DATA_PATH, "labels", "augmented"))
    except FileNotFoundError:
        print("There is no augmented train directory, creating it")
        os.makedirs(os.path.join(YOLO_DATA_PATH, "labels", "augmented"))

    # combination of the differents parameters of augmentation
    degree = [math.pi / 2, math.pi, 3 / 2 * math.pi]
    flipping_param = [1, 0, -1]
    combinaison = list(product(degree, flipping_param))

    yolo_image_train_path = os.path.join(YOLO_DATA_PATH, "images", "train")

    # computing how many image we need to augment per viruses
    nb_to_create = dictionary_initialization(yolo_image_train_path)

    # copy paste existing image
    for image in os.listdir(yolo_image_train_path):
        copyfile(
            os.path.join(yolo_image_train_path, image),
            os.path.join(YOLO_DATA_PATH, "images", "augmented", image),
        )
    for label in os.listdir(os.path.join(YOLO_DATA_PATH, "labels", "train")):
        copyfile(
            os.path.join(YOLO_DATA_PATH, "labels", "train", label),
            os.path.join(YOLO_DATA_PATH, "labels", "augmented", label),
        )

    # augmentation per viruses
    for key in nb_to_create:
        nbr_passage = 0
        while nb_to_create[key] > 0:
            degree = combinaison[nbr_passage][0]
            flip = combinaison[nbr_passage][1]
            for file in os.listdir(yolo_image_train_path):
                ind = file.find("_")
                virus = file[:ind]
                if virus == key:
                    # augementation de l'image
                    image_augmented(file, degree, flip, nbr_passage)
                    # écriture du document label
                    file_path = get_label_file(file)
                    new_label_documentation(file, file_path, degree, flip, nbr_passage)
                    nb_to_create[virus] = nb_to_create[virus] - 1
                    if nb_to_create[virus] == 0:
                        break
            nbr_passage += 1


if __name__ == "__main__":
    augmentation_of_yolo()
