from fastapi_chameleon import template
import fastapi

router = fastapi.APIRouter()


@router.get("/")
@template()
def index(welcome: str = 'here you will find general information'):
    return {
        'welcome': welcome
    }
@router.get("/index.pt")
@template()
def index(welcome: str = 'here you will find general information'):
    return {
        'welcome': welcome
    }


@router.get("/tools")
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
@template()
def description(welcome: str = 'here you will find general description'):
    return {
        'welcome': welcome
    }
@router.get("/description.pt")
@template()
def description(welcome: str = 'here you will find general description'):
    return {
        'welcome': welcome
    }

@router.get("/statistic")
@template()
def statistic(welcome: str = 'here you will find statistic'):
    return {
        'welcome': welcome
    }

@router.get("/statistic.pt")
@template()
def statistic(welcome: str = 'here you will find statistic'):
    return {
        'welcome': welcome
    }

@router.get("/contact")
@template()
def contact(welcome: str = 'here you will find contact'):
    return {
        'welcome': welcome
    }
@router.get("/contact.pt")
@template()
def contact(welcome: str = 'here you will find contact'):
    return {
        'welcome': welcome
    }

@router.get("/Defect")
@template()
def defect(welcome: str = 'here you will find defect'):
    return {
        'welcome': welcome
    }
@router.get("/Localization")
@template()
def localization(welcome: str = 'here you will find localization'):
    return {
        'welcome': welcome
    }

