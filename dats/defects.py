import keras._tf_keras.keras
import numpy as np
from keras._tf_keras.keras.preprocessing.image import img_to_array
from pathlib import Path
import os
import pandas as pd

class defect:
    def __init__(self, maintainers: list = None):
        self.score= list()
        self.id = list()
        self.file_stat = os.path.join(Path(os.getcwd()).parent.absolute(),
                                      'Severstal_API','stat_class.csv')
        self.file_count = len(os.listdir(os.path.join(
            Path(Path(os.getcwd()).parent.absolute(), 'Severstal_API/data'))))
        self.base_dir = os.path.join(
            Path(Path(os.getcwd()).parent.absolute(), 'data'))
        self.model_path = os.path.join(Path(os.getcwd()).parent.absolute(),
                                       'Severstal_API/models/resnet50.keras')
        self.dir = Path(os.getcwd()).parent.absolute()
        if maintainers is None:
            maintainers = list()
        self.maintainers = maintainers
        self.base_dir = os.path.join(Path(Path(os.getcwd()).parent.absolute(), 'Severstal_API/data'))
        self.model = keras.models.load_model(self.model_path)

    @property
    def res(self) -> list:
        for i in range(1, 17):
            name = os.path.join(self.base_dir, os.listdir
            (self.base_dir)
            [self.file_count - i])
            fn = keras.utils.load_img(name, target_size=(224, 224))
            x = img_to_array(fn).astype(
                "float32")  # Numpy array with shape (224,224, 3)
            x /= 255
            x = x.reshape(
                (1,) + x.shape)  # Numpy array with shape (1, 224,224, 3)
            image_test = np.vstack([x])

            klass = self.model.predict(image_test)
            if klass[0][0] >= 0.5:
                case = {'id': os.listdir(self.base_dir)[self.file_count - i],
                        'path': os.path.join('../../data', os.listdir(
                            self.base_dir)[self.file_count - i]),
                        'summary': "Detail is norm: " + str(np.round(klass[0],
                                                                     2))}
                self.maintainers.append(case)
                self.score.append(np.round(klass[0][0],2))
                self.id.append(os.listdir(self.base_dir)[self.file_count - i])
            else:
                case = {'id': os.listdir(self.base_dir)[self.file_count -
                                                        i],
                        'path': os.path.join('../../data', os.listdir(
                            self.base_dir)[self.file_count - i]), 'summary':
                            "Detail is defect: " + str(np.round(klass[0], 2))}
                self.maintainers.append(case)
                self.score.append(np.round(klass[0][0], 2))
                self.id.append(os.listdir(self.base_dir)[self.file_count - i])
        stat_class = {"id": self.id,"score": self.score}
        pd.DataFrame(stat_class).to_csv(self.file_stat)
        return self.maintainers
