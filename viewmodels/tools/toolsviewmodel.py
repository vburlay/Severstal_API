from typing import List
from services import classification_service,dowload_servises,segmentation_service
from viewmodels.shared.viewmodel import ViewModelBase
from starlette.requests import Request

class IndexViewModel(ViewModelBase):
    def __init__(self, request: Request):
        super().__init__(request)

        self.download: int = dowload_servises
        self.segmentation: int = segmentation_service
        self.classification: int = classification_service



