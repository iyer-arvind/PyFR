# -*- coding: utf-8 -*-

import uuid
from abc import ABCMeta, abstractmethod


class BaseReader(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _to_raw_pyfrm(self):
        pass

    def to_pyfrm(self):
        mesh = self._to_raw_pyfrm()

        # Add metadata
        mesh['mesh_uuid'] = uuid.uuid4().hex

        return mesh
