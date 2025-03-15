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


@router.get("/plots")
@router.get("/plots.pt")
@template()
def plots(welcome: str = 'here you will find visualization'):
    return {
        'welcome': welcome,
        'functions': [
            {'id': 'Plots', 'summary': 'Check whether the details are '
                                        'defective.'},
            {'id': 'Diagrams','summary':  'Check where selected details are located'}
        ]
    }
@router.get("/Plots", response_class=HTMLResponse)

async def surival_rate_plotly():
    df = pd.read_csv(os.path.join(Path(os.getcwd()).parent.absolute(),
                                   'Severstal_API', 'stat_class.csv'), sep=",")
    df["Defect"] = df["score"].map(lambda x: 'Normal' if x >= 0.5 else 'Defect')

    survival_rate = df[['Defect','id']]
    fig = px.histogram(survival_rate,x='Defect', color="Defect",
                 title='Count of defects (Model Classification)')
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

@router.get("/Diagrams", response_class=HTMLResponse)

async def surival_rate_plotly():
    df1 = pd.read_csv(os.path.join(Path(os.getcwd()).parent.absolute(),
                                  'Severstal_API', 'stat_loc.csv'), sep=",")[['id','max_pixel','defect']]


    df2 = pd.read_csv(os.path.join(Path(os.getcwd()).parent.absolute(),
                                   'Severstal_API', 'stat_class.csv'), sep=",")[['id','score']]

    df = pd.merge(df1, df2, on='id')
    survival_rate = df.groupby(['defect']).count().reset_index()
    fig = px.bar(survival_rate, x='defect', y= 'score' ,color='defect',
                 title='Count of defects (UNET-Model)', labels={'defect': 'Defect number', 'score':'Count'})
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
