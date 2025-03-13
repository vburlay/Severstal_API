from fastapi_chameleon import template
import fastapi
from starlette.requests import Request
from viewmodels.classification.defect import IndexViewModel_c
from viewmodels.localization.segmentation import IndexViewModel_l

from fastapi.responses import HTMLResponse
import plotly.express as px
from plotly.io import to_html
import pandas as pd
import os
from pathlib import Path

router = fastapi.APIRouter()


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


@router.get("/statistic", response_class=HTMLResponse)
@router.get("/statistic.pt", response_class=HTMLResponse)
@template()
async def surival_rate_plotly():
    df = pd.read_csv(os.path.join(Path(os.getcwd()).parent.absolute(),
                                  'Severstal_API', 'stat_loc.csv'), sep=",")
    survival_rate = df.groupby(['defect']).count().reset_index()
    fig = px.bar(survival_rate, x='defect', y='score',
                 title='Count of defects')
    plot_div = to_html(fig, full_html=False)
    html_content = f"""
            <html>
                <head>
                    <title>Survival Rate Plot</title>
                </head>
                <body>
                    {plot_div}
                </body>
            </html>

"""
    return HTMLResponse(content=html_content)


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
