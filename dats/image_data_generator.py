from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
import keras._tf_keras.keras
from dataclasses import dataclass
import os
import numpy as np
from pathlib import Path
import PIL.Image as Image
import pandas as pd

@dataclass
class G:
    img_height = 224
    img_width = 224
    nb_epochs = 100
    batch_size = 64
    BACKBONE = "resnet50"
    path = os.path.join(os.getcwd(), "data")
    train_dir = os.path.join(path, "ml_classification/")
    RGB = 3
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=5,
            mode="min",
            factor=0.2,
            min_lr=1e-7,
            verbose=1,
        )
    ]
def train_val_generators():
    gen_train = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
        preprocessing_function=preprocess_input,
    )
    generator_train = gen_train.flow_from_directory(
        directory=G.train_dir,
        target_size=(G.img_height, G.img_width),
        batch_size=G.batch_size,
        shuffle=True,
        color_mode="rgb",
        class_mode="binary",
        subset="training",
        seed=21,
    )
    generator_validation = gen_train.flow_from_directory(
        directory=G.train_dir,
        target_size=(G.img_height, G.img_width),
        batch_size=G.batch_size,
        shuffle=True,
        color_mode="rgb",
        class_mode="binary",
        subset="validation",
        seed=21,
    )
    return generator_train, generator_validation


def rle2maskResize(rle):
    # CONVERT RLE TO MASK
    if (pd.isnull(rle)) | (rle == ""):
        return np.zeros((128, 800), dtype=np.uint8)

    height = 256
    width = 1600
    mask = np.zeros(width * height, dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2] - 1
    lengths = array[1::2]
    for index, start in enumerate(starts):
        mask[int(start): int(start + lengths[index])] = 1

        return mask.reshape((height, width), order="F")[::2, ::2]


class DataGenerator(keras.utils.Sequence):
    def __init__(
            self, df, batch_size=16, subset="train", shuffle=False,
            preprocess=None, info={}
    ):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.info = info

        if self.subset == "train":
            self.data_path = G.path + "/train_images/"
        elif self.subset == "test":
            self.data_path = G.path + "/test_images/"
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        X = np.empty((self.batch_size, 128, 800, 3), dtype=np.float32)
        y = np.empty((self.batch_size, 128, 800, 4), dtype=np.int8)
        indexes = self.indexes[
                  index * self.batch_size: (index + 1) * self.batch_size]
        for i, f in enumerate(self.df["ImageId"].iloc[indexes]):
            self.info[index * self.batch_size + i] = f
            X[i,] = Image.open(self.data_path + f).resize((800, 128))
            if self.subset == "train":
                for j in range(4):
                    y[i, :, :, j] = rle2maskResize(
                        self.df["e" + str(j + 1)].iloc[indexes[i]]
                    )
        if self.preprocess != None:
            X = self.preprocess(X)
        if self.subset == "train":
            return X, y
        else:
            return X


def generators_unet():
    path = os.path.join(os.getcwd(), "data")
    train = pd.read_csv(path + "/train.csv")
    train2 = train.pivot(index="ImageId", columns="ClassId",
                         values="EncodedPixels")
    train2.fillna("", inplace=True)
    train2["count"] = np.sum(train2.iloc[:] != "", axis=1).values
    train2 = pd.DataFrame(train2.to_records())
    train2.rename(columns={"1": "e1", "2": "e2", "3": "e3", "4": "e4"},
                  inplace=True)

    idx = int(0.8 * len(train2))
    train_batches = DataGenerator(
        train2.iloc[:idx], shuffle=True, preprocess=G.preprocess_input
    )
    valid_batches = DataGenerator(train2.iloc[idx:],
                                  preprocess=G.preprocess_input)
    return train_batches, valid_batches
