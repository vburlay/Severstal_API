from typing import List
from services import segmentation_service
from viewmodels.shared.viewmodel import ViewModelBase
from starlette.requests import Request

class IndexViewModel_l(ViewModelBase):
    def __init__(self, request: Request):
        super().__init__(request)
        self.localization: List = segmentation_service.localization_image()
        self.welcome: str = 'here you will find localization of defects'
