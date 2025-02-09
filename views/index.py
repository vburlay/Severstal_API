from fastapi_chameleon import template
import fastapi

router = fastapi.APIRouter()


@router.get("/")
@template()
def index(welcome:str = 'here you will find general information'):
    return {
        'welcome': welcome
    }

@router.get("/tools")
@template()
def tools(welcome:str = 'here you will find functionality'):
    return {
        'welcome': welcome
    }

@router.get("/description")
@template()
def description(welcome:str = 'here you will find general description'):
    return {
        'welcome': welcome
    }

@router.get("/statistic")
@template()
def statistic(welcome:str = 'here you will find statistic'):
    return {
        'welcome': welcome
    }

@router.get("/contact")
@template()
def contact(welcome:str = 'here you will find contact'):
    return {
        'welcome': welcome
    }
