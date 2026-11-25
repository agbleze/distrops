# read version from installed package

from . import engine
from . import task



from importlib.metadata import version
__version__ = version("distrops")