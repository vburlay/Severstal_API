import keras._tf_keras.keras
from dataclasses import dataclass
import os
import numpy as np
import PIL.Image as Image

@dataclass
class G:
    path = os.path.join(os.getcwd(), "data/")

class DataGenerator(keras.utils.Sequence):
    def __init__(
            self, df, batch_size=16,  shuffle=False,
            preprocess=None, info={}
    ):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.info = info
        self.data_path = G.path
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
        if self.preprocess != None: X = self.preprocess(X)
        return (X,)


