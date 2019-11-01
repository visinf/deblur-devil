from utils import factories


def init():
    factories.import_submodules(__name__)
