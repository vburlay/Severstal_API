from typing import List
from services import classification_service
from viewmodels.shared.viewmodel import ViewModelBase
from starlette.requests import Request

class IndexViewModel_c(ViewModelBase):
    def __init__(self, request: Request):
        super().__init__(request)
        self.classification: List = classification_service.defect_image(
            limit=16)
        self.welcome: str = 'Here you will find defect if threshold < [0.5]'



