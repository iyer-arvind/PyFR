# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import re
import uuid

import numpy as np


class BaseReader(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _to_raw_pyfrm(self):
        pass

    def to_pyfrm(self):
        return self._to_raw_pyfrm()
