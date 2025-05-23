import fastapi
import uvicorn
import fastapi_chameleon
from starlette.staticfiles import StaticFiles
from views import index

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
