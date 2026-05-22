import os
from osgeo import ogr, gdal
import numpy as np


# ------------------------------------------------------------------------------------------------
# Zonal mean calculation
# ------------------------------------------------------------------------------------------------
# https://towardsdatascience.com/zonal-statistics-algorithm-with-python-in-4-steps-382a3b66648a
# https://medium.com/towards-data-science/zonal-statistics-algorithm-with-python-in-4-steps-382a3b66648a

FID_KEY = 'fid'
FIELD_KEY = 'field_key'
MIN_KEY = 'min'
MAX_KEY = 'max'
MEAN_KEY = 'mean'
MEDIAN_KEY = 'median'
STD_KEY = 'std'
SUM_KEY = 'sum'
COUNT_KEY = 'count'


def zonal_stats(fn_raster, fn_zones, key_field=None, nodata_value=0):
    def boundingBoxToOffsets(bbox, geot):
        col1 = int((bbox[0] - geot[0]) / geot[1])
        col2 = int((bbox[1] - geot[0]) / geot[1]) + 1
        row1 = int((bbox[3] - geot[3]) / geot[5])
        row2 = int((bbox[2] - geot[3]) / geot[5]) + 1
        return [row1, row2, col1, col2]

    def geotFromOffsets(row_offset, col_offset, geot):
        new_geot = [
            geot[0] + (col_offset * geot[1]),
            geot[1],
            0.0,
            geot[3] + (row_offset * geot[5]),
            0.0,
            geot[5]
        ]
        return new_geot

    mem_driver = ogr.GetDriverByName("Memory")
    mem_driver_gdal = gdal.GetDriverByName("MEM")
    shp_name = "temp"

    r_ds = gdal.Open(fn_raster)
    p_ds = ogr.Open(fn_zones)

    lyr = p_ds.GetLayer()
    geot = r_ds.GetGeoTransform()
    nodata_raster = r_ds.GetRasterBand(1).GetNoDataValue()

    zstats = []

    p_feat = lyr.GetNextFeature()

    while p_feat:
        if p_feat.GetGeometryRef() is not None:
            if os.path.exists(shp_name):
                mem_driver.DeleteDataSource(shp_name)
            tp_ds = mem_driver.CreateDataSource(shp_name)
            tp_lyr = tp_ds.CreateLayer('polygons', None, ogr.wkbPolygon)
            tp_lyr.CreateFeature(p_feat.Clone())
            offsets = boundingBoxToOffsets(p_feat.GetGeometryRef().GetEnvelope(),
                                           geot)
            new_geot = geotFromOffsets(offsets[0], offsets[2], geot)

            tr_ds = mem_driver_gdal.Create(
                "",
                offsets[3] - offsets[2],
                offsets[1] - offsets[0],
                1,
                gdal.GDT_Byte)

            tr_ds.SetGeoTransform(new_geot)
            gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values=[1])
            tr_array = tr_ds.ReadAsArray()

            r_array = r_ds.GetRasterBand(1).ReadAsArray(
                offsets[2],
                offsets[0],
                offsets[3] - offsets[2],
                offsets[1] - offsets[0])

            fid = p_feat.GetFID()

            key_filed_value = nodata_value
            if key_field is not None:
                key_filed_value = p_feat.GetField(key_field)

            fstats = {
                FID_KEY: fid,
                FIELD_KEY: key_filed_value,
                MIN_KEY: nodata_value,
                MAX_KEY: nodata_value,
                MEAN_KEY: nodata_value,
                MEDIAN_KEY: nodata_value,
                STD_KEY: nodata_value,
                SUM_KEY: nodata_value,
                COUNT_KEY: nodata_value,
            }

            if r_array is not None:
                maskarray = np.ma.MaskedArray(
                    r_array,
                    mask=np.logical_or(r_array == nodata_raster, np.logical_not(tr_array)))

                if maskarray is not None:
                    fstats[MIN_KEY] = maskarray.min()
                    fstats[MAX_KEY] = maskarray.max()
                    fstats[MEAN_KEY] = maskarray.mean()
                    fstats[MEDIAN_KEY] = np.ma.median(maskarray)
                    fstats[STD_KEY] = maskarray.std()
                    fstats[SUM_KEY] = maskarray.sum()
                    fstats[COUNT_KEY] = maskarray.count()

            zstats.append(fstats)
            p_feat = lyr.GetNextFeature()
    return zstats