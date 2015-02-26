from .pyfrio import BaseIO
from .h5pyio import H5FileIO

from pyfr.util import subclass_where,subclasses

def get_io_by_name(name, *args, **kwargs):
    return subclass_where(BaseIO, name=name)(*args, **kwargs)

def get_io_by_extn(extn, *args, **kwargs):
    io_map = {ex: cls
                  for cls in subclasses(BaseIO)
                  for ex in cls.extn}

    return io_map[extn](*args, **kwargs)
