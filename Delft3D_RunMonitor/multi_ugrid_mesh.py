import numpy as np
import pyvista as pv

from .ugrid_mesh import UGridMesh
import time


class MultiUGridMesh(UGridMesh):
    """
    A class that combines data and grid stored in mutiple partitions
    """

    def __init__(self, filenames):
        """
        Constructor

        :param filenames: list of map filenames
        """
        self.meshes = [UGridMesh(fn) for fn in filenames]
        self.time = 0
        if len(self.meshes) > 0:
            self.time = self.meshes[0].time

    def readField(self, varname: str, time_index: int):
        """
        Read the field values at time time_index from the NetCDF file

        :param varname: variable name
        :param time_index: time index
        """
        data_list = [m.readField(varname=varname, time_index=time_index) for m in self.meshes]
        return np.concatenate(data_list)


    def to_pyvista(self, varname=None, time_index=None):
        """
        Convert mesh to a PyVista PolyData object

        :param varname: variable name
        :param time_index: time index
        """
        polydata = pv.merge([m.to_pyvista(varname, time_index) for m in self.meshes])
        return polydata

