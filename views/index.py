from fastapi_chameleon import template
import fastapi
from starlette.requests import Request
from viewmodels.classification.defect import IndexViewModel_c
from viewmodels.localization.segmentation import IndexViewModel_l

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
import plotly.express as px
from plotly.io import to_html
import seaborn as sns
import pandas as pd
import os
from pathlib import Path

router = fastapi.APIRouter()

# df = pd.read_csv(os.path.join(Path(os.getcwd()).parent.absolute(),
#                                    'Severstal_API', 'stat_loc.csv'),sep=",")
# survival_rate = df.groupby(['defect']).count().reset_index()


@router.get("/")
@router.get("/index.pt")
@template()
def index(welcome: str = 'here you will find general information'):
    return {'welcome': welcome}


@router.get("/tools")
@router.get("/tools.pt")
@template()
def tools(welcome: str = 'here you will find functionality'):
    return {
        'welcome': welcome,
        'functions': [
            {'id': 'Defect', 'summary': 'Check whether the details are '
                                        'defective.'},
            {'id': 'Localization',
             'summary': 'Check where selected details are located'}
        ]
    }


@router.get("/description")
@router.get("/description.pt")
@template()
def description(welcome: str = 'here you will find general description'):
    return {'welcome': welcome}


@router.get("/statistic")
@router.get("/statistic.pt")
@template()
def statistic(welcome: str = 'here you will find statistic'):
    # fig = px.bar(survival_rate, x='defect', y='count',
    #              title='Count of defects')

    return {'welcome': welcome}


@router.get("/contact")
@router.get("/contact.pt")
@template()
def contact(welcome: str = 'here you will find contact'):
    return {'welcome': welcome}


@router.get("/Defect")
@template()
def defect(request: Request):
    vm = IndexViewModel_c(request)
    return vm.to_dict()


@router.get("/Localization")
@template()
def localization(request: Request):
    vm = IndexViewModel_l(request)
    return vm.to_dict()
