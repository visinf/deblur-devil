# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

from utils import factories


def init():
    factories.import_submodules(__name__)
