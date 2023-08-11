from importlib.metadata import version

from .modules import DrugBankConnector

__all__ = ["modules"]

__version__ = version("drugbankpy")
