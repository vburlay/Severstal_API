# Steel Defect Detection

![image](/static/bilder/CHMF.ME_BIG.png ) 

>API based on FastAPI.

## Table of Contents 

* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Contact](#contact)



## General Information

> To improve the efficiency of steel production, this program will help engineers automate the process of locating and classifying surface defects on steel plate.
> This program uses the following algorithms:
 > - Classification (binary ResNet50-model).
 > - Segmentation (U-Net).

 > Dataset: "Severstal" comes from Kaggle [_here_](https://www.kaggle.com/competitions/severstal-steel-defect-detection/overview).

## Technologies Used
- Python - version 3.10
- FastAPI
- Docker
- AWS ECS


## Features
- fastapi_chameleon,uvicorn,fastapi,starlette
- HTML,CSS

## Screenshots
- WebAPI

![Example screenshot](/static/bilder/api.png)

- Defect-classification

![Example screenshot](/static/bilder/classification.png)

- Localization

![Example screenshot](/static/bilder/localization.png)



## Setup

It is necessary to install the following Python-Packages additionally: 
```r
viewmodel,fastapi_chameleon,uvicorn,fastapi,starlette
```

## Usage

* Preparation
```r
app = fastapi.FastAPI()

def main():
    configure()
    uvicorn.run(app, host="127.0.0.1", port=8000)

def configure():
    configure_templates()
    configure_routes()

def configure_templates():
    fastapi_chameleon.global_init('templates')

def configure_routes():
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.mount("/data", StaticFiles(directory="data"), name="data")

    app.include_router(index.router)

if __name__ == "__main__":
    main()
else:
    configure()

```
* Plots
```r
@router.get("/Histogram", response_class=HTMLResponse)

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
```                 


## Project Status

Project is complete 


## Room for Improvement

* By using JavaScript, it could be better interface achieved.