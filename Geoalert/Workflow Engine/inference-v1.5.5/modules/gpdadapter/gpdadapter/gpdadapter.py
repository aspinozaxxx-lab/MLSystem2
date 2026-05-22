import geopandas as gpd
import pandas as pd
from typing import (Union, Optional, Callable, Sequence, Any, List, Tuple, Final, Hashable, Literal, TypeVar,
                    Dict, Iterable)
import numpy as np
from pyproj import CRS
from shapely.geometry.base import BaseGeometry
import shapely
import warnings
from loguru import logger

GEOMETRY: Final[str] = 'geometry'
# acceptable types of the first index before parsing
ROWS_INDEX_TYPE = Union[int, slice, np.ndarray, np.ndarray, List[int], pd.Series]
PARSED_ROWS_INDEX_TYPE = Union[int, np.ndarray]  # types of the first index after parsing
COLUMNS_INDEX_TYPE = Union[str, Sequence[str]]
INIT_DATA_TYPE = Union[dict, Sequence[dict], Sequence[pd.Series], List[BaseGeometry],
                       gpd.GeoDataFrame, BaseGeometry, None]
Self = TypeVar("Self", bound="FeatureCollection")
DEFAULT_CRS: Final[str] = 'EPSG:4326'


class FeatureCollection:
    def __init__(self,
                 data: INIT_DATA_TYPE = None,
                 crs: Union[CRS, str] = DEFAULT_CRS):
        """
        data:
            None -> empty FeatureCollection
            GeoDataFrame -> _data = data, crs = data.crs
            Series -> will have single row and columns from data.index (must have 'geometry')
            dict[non Sequences] -> will have single row and columns from data.keys (must have 'geometry')
            dict[Sequences] -> will have columns from data.keys (must have 'geometry')
            Sequence[geometries] -> will have single 'geometry' column
            GeoSeries -> will have single 'geometry' column and inherit crs
            Sequence[dict, Series] -> concat row-wise
            Sequence[FeatureCollections] -> concat row-wise inherit crs
        """
        if data is None or (isinstance(data, (Sequence, dict, pd.Series, gpd.GeoSeries)) and len(data) == 0):  # empty
            self._data = gpd.GeoDataFrame(geometry=[])
            return

        if isinstance(data, BaseGeometry):  # single geometry
            data = [data]

        # From sequence of rows ----------------------------------------------------------------------------------------
        if isinstance(data, Sequence):
            if all(isinstance(d, BaseGeometry) for d in data):  # sequence of geometries
                data = {GEOMETRY: data} if len(data) > 0 else None
            elif all(isinstance(d, (dict, pd.Series)) for d in data):  # sequence of dicts or Series(rows)
                data = gpd.GeoDataFrame(data, geometry=GEOMETRY, crs=crs)
            elif all(isinstance(d, FeatureCollection) for d in data):   # sequence of FeatureCollections
                data = pd.concat((d._data for d in data), ignore_index=True)
            else:
                raise ValueError(f'Can not parse {data}')

        if isinstance(data, gpd.GeoSeries):    # sequence of geometries as GeoSeries with crs
            data = {GEOMETRY: data} if len(data) > 0 else None

        # From dict of columns -----------------------------------------------------------------------------------------
        if isinstance(data, pd.Series) and not isinstance(data, gpd.GeoSeries):  # single Series(row)
            data = data.to_dict()
        if isinstance(data, dict):
            data = {k: v if isinstance(v, (Sequence, gpd.GeoSeries, pd.Series, np.ndarray)) else [v] for k, v in
                    data.items()}
            data = gpd.GeoDataFrame(data, geometry=GEOMETRY,
                                    crs=data[GEOMETRY].crs if isinstance(data[GEOMETRY], gpd.GeoSeries) else crs)

        if isinstance(data, pd.DataFrame) and not isinstance(data, gpd.GeoDataFrame):  # from DataFrame
            if GEOMETRY not in data.columns:
                data[GEOMETRY] = gpd.GeoSeries()
            data = gpd.GeoDataFrame(data, geometry=GEOMETRY, crs=crs)

        if len(data) == 0:   # TODO: refactor
            data[GEOMETRY] = pd.Series(dtype=object)

        # at this point data must be GeoDataFrame
        assert isinstance(data, gpd.GeoDataFrame) and hasattr(data, GEOMETRY)
        _crs = data.crs if len(data) > 0 else None
        if len(data) == 0:
            data.set_crs(None, inplace=True, allow_override=True)
        if len(data) > 0 and not _crs:
            warnings.warn('Got undefined CRS, assigning default EPSG:4326')
            data.set_crs(crs, inplace=True, allow_override=True)
        self._data = data
        self._data.reset_index(drop=True, inplace=True)
        self.validate()



    # INTERNAL METHODS -------------------------------------------------------------------------------------------------
    @staticmethod
    def _parse_index(index: Union[Tuple[ROWS_INDEX_TYPE, COLUMNS_INDEX_TYPE],
                            ROWS_INDEX_TYPE]) -> Tuple[ROWS_INDEX_TYPE, Optional[COLUMNS_INDEX_TYPE]]:
        if isinstance(index, tuple):  # got rows and columns
            if len(index) != 2:
                raise IndexError(f"Expecting 2d index, got {index}")
            return index[0], index[1]
        else:  # got rows only
            return index, None

    #@staticmethod
    def _parse_rows(self, rows: ROWS_INDEX_TYPE) -> PARSED_ROWS_INDEX_TYPE:
        if isinstance(rows, pd.Series):  # indexing with boolean condition
            return rows.values
        if isinstance(rows, List):
            return np.array(rows)
        if isinstance(rows, slice):
            return np.arange(rows.start or 0, rows.stop or len(self._data), rows.step or 1)
        if isinstance(rows, int):
            return rows
        return rows

    def _column_iloc(self, columns: COLUMNS_INDEX_TYPE) -> Union[int, Sequence[int]]:
        """Returns integer index or list of indexes for column name or list of names"""
        if isinstance(columns, str):
            return self._data.columns.get_loc(columns)
        elif isinstance(columns, (list, tuple)):
            return [self._data.columns.get_loc(c) for c in columns]

    @staticmethod
    def _infer_value_dtype(value: Any) -> type:
        if isinstance(value, (np.ndarray, pd.Series)):
            return value.dtype
        if isinstance(value, list):
            return float if all(type(i) in (int, float) for i in value) else object
        if isinstance(value, int):
            return int
        if isinstance(value, float):
            return float
        return object

    def validate(self):
        if self._data.empty:
            return
        self._data.geometry.map(FeatureCollection._validate_geometry)

    @staticmethod
    def _validate_geometry(geometry: BaseGeometry) -> BaseGeometry:
        # TODO: Check geometries type (Polygons only), filter out empty geometries, etc.
        return geometry

    # MAGIC METHODS ----------------------------------------------------------------------------------------------------
    def __bool__(self):
        return not self._data.empty

    def __repr__(self):
        return self._data.__repr__().replace('GeoDataFrame', 'FeatureCollection')

    def __str__(self):
        return self._data.__str__().replace('GeoDataFrame', 'FeatureCollection')

    def __len__(self):
        return len(self._data)

    @property
    def loc(self):
        """For compatibility with DataFrame, not recommended for usage"""
        return self._data.loc

    @property
    def iloc(self):
        """For compatibility with DataFrame, not recommended for usage"""
        return self._data.iloc

    @property
    def at(self):
        """For compatibility with DataFrame, not recommended for usage"""
        return self._data.at

    @property
    def iat(self):
        """For compatibility with DataFrame, not recommended for usage"""
        return self._data.iat

    def __getitem__(self, index: Union[Tuple[ROWS_INDEX_TYPE, COLUMNS_INDEX_TYPE], ROWS_INDEX_TYPE]) -> Any:
        """
        Always returns a copy
        index must be Tuple[rows, columns] or rows;
        rows can be row indexes as int, slice, List[int], array[int]
        or boolean indexes (condition) as array[bool] or Series[bool];
        columns is a column names, str or Sequence[str];
        Returns single value, Series, DataFrame or FeatureCollection depending on item
        """
        rows, columns = self._parse_index(index)
        result = self._data.iloc[self._parse_rows(rows), self._column_iloc(columns)] if columns is not None else \
            self._data.iloc[self._parse_rows(rows)]
        if isinstance(result, gpd.GeoSeries) or isinstance(result, gpd.GeoDataFrame) or (
                isinstance(result, pd.Series) and GEOMETRY in result.index):
            result = FeatureCollection(result, crs=self.crs)
        return result

    def __setitem__(self, index: Union[Tuple[ROWS_INDEX_TYPE, COLUMNS_INDEX_TYPE], ROWS_INDEX_TYPE], value: Any):
        """
        index must be Tuple[rows, columns] or rows;
        rows can be row indexes as int, slice, List[int], array[int]
        or boolean indexes (condition) as array[bool] or Series[bool];
        columns must be a single column name
        Value must be a single value or sequence when setting column or one of {dict, FeatureCollection, Series}
        when setting rows
        """
        rows, column = self._parse_index(index)
        rows = self._parse_rows(rows)
        if column is None:
            # column is None -> assigning the whole row
            if isinstance(value, FeatureCollection):
                self._data.iloc[rows] = value._data
            else:
                self._data.iloc[rows] = value
            return

        if column in self._data.columns:
            # existing column
            column = self._column_iloc(column)
            if isinstance(rows, int):  # single row -> can assign Iterable value
                self._data.iat[rows, column] = value
            else:  # multiple rows
                if hasattr(value, '__len__') and not isinstance(value, str):  #  if value is an Iterable checking its length
                    if len(value) == 0:
                        self._data.iloc[rows, column] = pd.Series([None]*len(rows),
                                                                  dtype=value.dtype if hasattr(value, 'dtype') else 'object')
                    elif len(value) != len(rows):
                        raise ValueError(f'Invalid assignment {value} to {column}, {rows}')
                    else:
                        self._data.iloc[rows, column] = value
                else:
                    self._data.iloc[rows, column] = value

        else:
            # new column
            if isinstance(rows, int) and rows == 0 and len(self._data) == 1:  #fc with a single row
                self._data[column] = value
            if isinstance(rows, np.ndarray) and len(rows) == len(self._data) and np.all(rows==np.arange(0, len(self._data))):
                self._data[column] = value
            else:
                self._data[column] = pd.Series(dtype=self._infer_value_dtype(value))
                if isinstance(rows, int):  # single row -> can assign Iterable value
                    self._data.iat[rows, self._data.columns.get_loc(column)] = value
                else:  # multiple rows -> if value is Iterable this will raise Exception
                    self._data.iloc[rows, self._data.columns.get_loc(column)] = value

    # CHECK AND VALIDATE DATA ------------------------------------------------------------------------------------------
    # noinspection PyTypeChecker
    def dropna(self,
               how: Literal['any', 'all'] = 'all',
               subset: Union[Hashable, Sequence[Hashable], None] = None,
               inplace: bool = False) -> Union[Self, None]:
        """Remove missing values.
        Args:
            how: 'any' - If any NA values are present, drop that row or column.
                 'all' - If all values are NA, drop that row or column.
            subset: column names
            inplace: inplace
        """
        if inplace:
            self._data.dropna(axis=0, how=how, subset=subset, inplace=inplace)
        else:
            return FeatureCollection(self._data.dropna(axis=0, how=how, subset=subset, inplace=inplace))

    def fillna(self,
               value: Any = None,
               method: Union[Literal['bfill', 'ffill'], None] = None,
               inplace: bool = False,
               limit: Union[int, None] = None) -> Union[Self, None]:
        """Fill NA/NaN values using the specified method.
        Args:
            value: Value to use to fill holes (e.g. 0), alternately a dict/Series/DataFrame of values
                   specifying which value to use for each index (for a Series) or column (for a DataFrame).
                   Values not in the dict/Series/DataFrame will not be filled. This value cannot be a list.
            method: Method to use for filling holes in reindexed Series, one of
                    (ffill: propagate last valid observation forward to next valid OR
                     bfill: use next valid observation to fill gap.)
            inplace: If True, fill in-place.
            limit: If method is specified, this is the maximum number of consecutive values to forward/backward fill.
        """
        if inplace:
            if method == 'bfill':
                self._data.bfill(inplace=True, limit=limit)
            elif method == 'ffill':
                self._data.ffill(inplace=True, limit=limit)
            else:
                self._data.fillna(value=value, axis=0, inplace=inplace, limit=limit)
        else:
            if method == 'bfill':
                data = self._data.copy()
                data.bfill(inplace=True, limit=limit)
                return FeatureCollection(data)
            elif method == 'ffill':
                data = self._data.copy()
                data.ffill(inplace=True, limit=limit)
                return FeatureCollection(data)
            return FeatureCollection(self._data.fillna(value=value, axis=0, inplace=inplace, limit=limit))

    def isna(self) -> pd.DataFrame:
        """Return a boolean same-sized object indicating if the values are NA."""
        return self._data.isna()

    def notna(self) -> pd.DataFrame:
        """Return a boolean same-sized object indicating if the values are not NA."""
        return self._data.notna()

    @property
    def is_empty(self) -> pd.Series:
        """Returns a Series of dtype('bool') with value True for empty geometries."""
        if self._data.empty:
            return pd.Series(dtype=bool)
        return self._data.is_empty

    @property
    def is_valid(self) -> pd.Series:
        """Returns a Series of dtype('bool') with value True for geometries that are valid."""
        if self._data.empty:
            return pd.Series(dtype=bool)
        return self._data.is_valid

    @property
    def is_valid_reason(self) -> pd.Series:
        """Returns a Series of strings with the reason for invalidity of each geometry."""
        if self._data.empty:
            return pd.Series(dtype=str)
        return pd.Series(shapely.is_valid_reason(self._data.geometry.values))

    @property
    def empty(self) -> bool:
        """Indicator whether FeatureCollection is empty."""
        return self._data.empty

    # ------------------------------------------------------------------------------------------------------------------
    def copy(self, deep: bool = True) -> Self:
        """Returns a copy of FeatureCollection"""
        return FeatureCollection(self._data.copy(deep))

    # APPLYING FUNCTIONS -----------------------------------------------------------------------------------------------
    def apply(self,
              func: Callable[[pd.Series], Any],
              axis: int = 1,
              raw: bool = False,
              result_type: Union[Literal['expand', 'reduce', 'broadcast'], None] = None,
              args=(), **kwargs) -> Union[pd.Series, pd.DataFrame, gpd.GeoSeries, gpd.GeoDataFrame]:
        """
        Args:
            func: Function to apply to each column or row.
            axis: 0 or 'index': apply function to each column. 1 or 'columns': apply function to each row.
            raw: Determines if row or column is passed as a Series or ndarray object
            result_type: These only act when axis=1 (columns):
                        'expand' : list-like results will be turned into columns.
                        'reduce' : returns a Series if possible rather than expanding list-like results.
                        'broadcast' : results will be broadcast to the original shape of the DataFrame,
                                      the original index and columns will be retained.
                        The default behaviour (None) depends on the return value of the applied function:
                        list-like results will be returned as a Series of those.
                        However, if the apply function returns a Series these are expanded to columns.
            args: Positional arguments to pass to func in addition to the array/series.
            kwargs: Additional keyword arguments to pass as keywords arguments to func.
        Returns:
            empty Series if empty"""
        # TODO: should we wrap into a FeatureCollection when result is DataFrame or GeoDataFrame?
        if self._data.empty:
            return pd.Series()
        return self._data.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwargs)

    def map(self, func: Callable[[Any], Any],
            column: str, na_action=None, inplace: bool = False) -> Union[pd.Series, None]:
        """Map to column"""
        if self._data.empty:
            return pd.Series() if not inplace else None
        if inplace:
            self._data[column] = self._data[column].map(func, na_action=na_action)
        else:
            return self._data[column].map(func, na_action=na_action)

    def filter(self,
               func: Callable[[pd.Series], bool],
               raw: bool = False,
               inplace: bool = False,
               args=(), **kwargs) -> Optional[Self]:
        """
        Args:
            func: Function to apply to each row. Must accept pd.Series and return bool.
            raw: Determines if row or column is passed as a Series or ndarray object
            inplace: inplace
            args: Positional arguments to pass to func in addition to the array/series.
            kwargs: Additional keyword arguments to pass as keywords arguments to func."""
        if self._data.empty:
            return self if not inplace else None
        if inplace:
            self._data = self._data.iloc[self._data.apply(func, axis=1, raw=raw, result_type='reduce',
                                                          args=args, **kwargs).values]
            self._data.reset_index(drop=True, inplace=True)
        else:
            data = self._data.iloc[self._data.apply(func, axis=1, raw=raw, result_type='reduce',
                                                    args=args, **kwargs).values]
            data.reset_index(drop=True, inplace=True)
            return FeatureCollection(data)

    # GENERAL MODIFICATIONS --------------------------------------------------------------------------------------------
    def sort(self, by: str, reverse: bool = False, inplace: bool = True,
             key: Optional[Callable] = None) -> Optional[Self]:
        if self._data.empty:
            return self if not inplace else None
        if inplace:
            self._data.sort_values(by=by, ascending=not reverse, inplace=True, key=key)
            self._data.reset_index(drop=True, inplace=True)
        else:
            return FeatureCollection(self._data.sort_values(by=by, ascending=not reverse, inplace=False, key=key))

    def append(self, other: Union[dict, pd.Series, Self]):
        """Append dict, Series or FeatureCollection inplace"""
        if isinstance(other, FeatureCollection):
            other = other._data
        if isinstance(other, dict):
            other = pd.DataFrame(other, index=[0])
        self._data = pd.concat((self._data, other), join='outer', ignore_index=True)

    def drop(self,
             labels: Union[Hashable, Sequence[Hashable], None] = None,
             axis: Union[Literal["index", "columns", "rows"], int] = 0,
             inplace: bool = False,
             errors: Literal["ignore", "raise"] = "ignore") -> Union[Self, None]:
        """Drop specified labels from rows or columns.
        Args:
            labels: Index or column labels to drop.
                    A tuple will be used as a single label and not treated as a list-like.
            axis: Whether to drop labels from the index (0 or 'index') or columns (1 or 'columns').
            inplace: If False, return a copy. Otherwise, do operation inplace and return None.
            errors: If 'ignore', suppress error and only existing labels are dropped.
            """
        if inplace:
            self._data.drop(labels=labels, axis=axis, inplace=inplace, errors=errors)
            if axis in (0, 'index', 'rows'):
                self._data.reset_index(drop=True, inplace=True)
        else:
            data = self._data.drop(labels=labels, axis=axis, inplace=inplace, errors=errors)
            if axis in (0, 'index', 'rows'):
                data.reset_index(drop=True, inplace=True)
            return FeatureCollection(data)

    # RTREE indexing ---------------------------------------------------------------------------------------------------
    def query(self,
              geometry: Union[BaseGeometry, Sequence[BaseGeometry], gpd.GeoSeries, Self],
              predicate: str = 'intersects') -> np.array:
        """Query rtree.
        Args:
            geometry: query geometry or sequence of geometries
            predicate: “contains”, “contains_properly”, “covered_by”, “covers”,
                       “crosses”, “intersects”, “overlaps”, “touches”, “within”
        Returns:
            ndarray with shape (n,) if geometry is a scalar Integer indices for matching geometries from the spatial
            index tree geometries or ndarray with shape (2, n) if geometry is an array_like. The first subarray
            contains input geometry integer indices. The second subarray contains tree geometry integer indices.
        """
        if self._data.empty:
            return np.array([])
        if isinstance(geometry, FeatureCollection):
            geometry = geometry._data.geometry
        if isinstance(geometry, gpd.GeoDataFrame):
            geometry = geometry.geometry
        return self._data.sindex.query(geometry, predicate=predicate)

    def clip(self, mask, keep_geom_type=False, inplace: bool = False) -> Optional[Self]:
        """Clip points, lines, or polygon geometries to the mask extent.
        Both layers must be in the same Coordinate Reference System (CRS).
        The GeoDataFrame will be clipped to the full extent of the mask object.
        If there are multiple polygons in mask, data from the GeoDataFrame will be clipped to the total
        boundary of all polygons in mask.
        Args:
            mask: Polygon vector layer used to clip the GeoDataFrame.
                  The mask's geometry is dissolved into one geometric feature and intersected with GeoDataFrame.
                  If the mask is list-like with four elements (minx, miny, maxx, maxy), clip will use a faster
                  rectangle clipping (~GeoSeries.clip_by_rect), possibly leading to slightly different results.
            keep_geom_type: If True, return only geometries of original type in case of intersection resulting
                            in multiple geometry types or GeometryCollections. If False, return all resulting
                            geometries (potentially mixed types).
            inplace: inplace
        """
        if self._data.empty:
            return self if not inplace else None
        if inplace:
            self._data = self._data.clip(mask, keep_geom_type)
        else:
            return FeatureCollection(self._data.clip(mask, keep_geom_type))

    def overlay(self, other: Union[BaseGeometry, Sequence[BaseGeometry], gpd.GeoSeries, Self],
                how: str = 'intersection', keep_geom_type: bool = True,
                make_valid: bool = True, inplace: bool = False) -> Optional[Self]:
        """Perform spatial overlay between GeoDataFrames.
        Args:
            other: GeoDataFrame
            how: Method of spatial overlay: 'intersection', 'union', 'identity',
                 'symmetric_difference' or 'difference'.
            keep_geom_type: If True, return only geometries of the same geometry type the GeoDataFrame has
            make_valid: If True, any invalid input geometries are corrected with a call to buffer(0)
            inplace: inplace
        """
        if isinstance(other, BaseGeometry):
            other = [other]
        if isinstance(other, Sequence) and all(isinstance(g, BaseGeometry) for g in other):
            # warnings.warn(f"Geometry passed to overlay, assuming crs is {self._data.crs}")
            other = gpd.GeoDataFrame(geometry=other, crs=self._data.crs if not self._data.empty else DEFAULT_CRS)
        if isinstance(other, gpd.GeoSeries):
            other = gpd.GeoDataFrame(geometry=other)
        if isinstance(other, FeatureCollection):
            other = other._data

        if self._data.empty and other.empty:  # both empty
            return self if not inplace else None

        if self._data.empty or other.empty:   # one of two is empty
            if how in ('union', 'difference', 'symmetric_difference'):
                overlay_result = other if self._data.empty else self._data
            else:
                return self

        else:  # both are OK
            overlay_result = self._data.overlay(other, how=how, keep_geom_type=keep_geom_type, make_valid=make_valid)

        if inplace:
            self._data = overlay_result
        else:
            return FeatureCollection(overlay_result)

    # IO ---------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_file(cls,
                  filename: Any,
                  bbox: Union[Tuple, BaseGeometry, gpd.GeoSeries, gpd.GeoDataFrame, None] = None,
                  mask: Union[dict, BaseGeometry, gpd.GeoSeries, gpd.GeoDataFrame, None] = None,
                  rows: Union[int, Sequence[int], slice, None] = None
                  ):
        """
        Args:
            filename: Either the absolute or relative path to the file or URL to be opened,
                      or any object with a read() method (such as an open file or StringIO)
            bbox: Filter features by given bounding box, GeoSeries, GeoDataFrame or a shapely geometry.
                  With engine="fiona", CRS mis-matches are resolved if given a GeoSeries or GeoDataFrame.
                  With engine="pyogrio", bbox must be in the same CRS as the dataset.
                  Tuple is (minx, miny, maxx, maxy) to match the bounds property of shapely geometry objects.
                  Cannot be used with mask.
            mask: Filter for features that intersect with the given dict-like geojson geometry,
                  GeoSeries, GeoDataFrame or shapely geometry. CRS mis-matches are resolved if given a GeoSeries
                  or GeoDataFrame. Cannot be used with bbox.
            rows: Load in specific rows by passing an integer (first n rows) or a slice() object.
        """
        return cls(gpd.read_file(filename=filename, bbox=bbox, mask=mask, rows=rows))

    def to_file(self, filename, hold_crs=False, driver: Any = None, schema: Any = None,
                index: Union[bool, None] = False, **kwargs):
        """Args:
            filename: File path or file handle to write to. The path may specify a GDAL VSI scheme.
            hold_crs: If True, save file in current CRS, else reproject to EPSG:4326
            driver: The OGR format driver used to write the vector file.
                    If not specified, it attempts to infer it from the file extension.
                    If no extension is specified, it saves ESRI Shapefile to a folder.
            schema: If specified, the schema dictionary is passed to Fiona to better control how the file is written.
                    If None, GeoPandas will determine the schema based on each column's dtype.
                    Not supported for the "pyogrio" engine.
            index: If True, write index into one or more columns (for MultiIndex).
                   Default None writes the index into one or more columns only if the index is named,
                   is a MultiIndex, or has a non-integer data type. If False, no index is written.
            kwargs: Keyword args to be passed to the engine"""
        data = self._data
        if data.empty:
            if not schema:  # need schema for empty gdf, otherwise fiona.to_file() will fail
                schema = {"geometry": "Polygon", "properties": {"id": "int"}}
            gpd.GeoDataFrame(geometry=[]).to_file(filename, driver=driver, schema=schema, index=index,
                                                  crs='EPSG:4326', engine='fiona')
            return
        if not hold_crs:
            data = data.to_crs('EPSG:4326')
        data.to_file(filename, driver=driver, schema=schema, index=index, engine='fiona', **kwargs)

    @classmethod
    def from_features(cls, features: Iterable, crs: Union[str, CRS], columns: Optional[Any] = None):
        try:
            return cls(gpd.GeoDataFrame.from_features(features, crs=crs, columns=columns))
        except ValueError as e:
            logger.warning(str(e))
            return cls()

    def to_json_str(self, **kwargs) -> Optional[str]:
        """Returns json string representation"""
        if self._data.empty:
            return '{"type": "FeatureCollection", "features": []}'
        return self._data.to_json(**kwargs)

    @classmethod
    def from_json_str(cls, geojson_str):
        try:
            return cls(gpd.read_file(geojson_str, driver='GeoJSON'))
        except ValueError as e:
            logger.warning(str(e))
            return cls()


    def to_json_dict(self, na="null", show_bbox=False, drop_id=False) -> Optional[dict]:
        """Returns json dict representation"""
        if self._data.empty:
            features = list()
            show_bbox = False
        else:
            features = list(self._data.iterfeatures(na=na, show_bbox=show_bbox, drop_id=drop_id))
        geo = {"type": "FeatureCollection",
               "features": features}
        if show_bbox:
            geo["bbox"] = tuple(self._data.total_bounds)
        return geo

    def plot(self, boundary: bool = False, column: Optional[str] = None, kind: str = 'geo', cmap: Optional[str] = None,
             color: Optional[str] = None, ax: Any = None, cax: Any = None, categorical: bool = False,
             legend: bool = False, vmin: Optional[float] = None, vmax: Optional[float] = None,
             markersize: Any = None, figsize: Tuple[int, int] = None, legend_kwds: Optional[dict] = None,
             categories: Any = None, aspect: str = 'auto', autolim: bool = True, **kwargs):
        """Plot via matplotlib. If a column is specified, the plot coloring will be based on values in that column.
        Args:
            boundary: if true, plot only polygons boundaries without filling
            column: The name of the dataframe column, array, or pd.Series to be plotted.
                    If array or pd.Series are used then it must have same length as dataframe.
                    Values are used to color the plot. Ignored if color is also set.
            kind: 'geo'-polygons, ‘line’-line plot, ‘bar’-vertical bar plot, ‘barh’-horizontal bar plot,
                  ‘hist’-histogram, ‘box’-BoxPlot, ‘kde’-Kernel Density Estimation plot, ‘area’-area plot
                  ‘pie’-pie plot, ‘scatter’-scatter plot, ‘hexbin’-hexbin plot.
            cmap: The name of a colormap recognized by matplotlib.
            color: If specified, all objects will be colored uniformly.
            ax: axes on which to draw the plot
            cax: axes on which to draw the legend in case of color map.
            categorical: If False, cmap will reflect numerical values of the column being plotted.
                         For non-numerical columns, this will be set to True.
            legend: Plot a legend. Ignored if no column is given, or if color is given.
            vmin: Minimum value of cmap. If None, the minimum data value in the column to be plotted is used.
            vmax: Maximum value of cmap. If None, the maximum data value in the column to be plotted is used.
            markersize: Point size
            figsize: Size of the Figure.
            legend_kwds: Keyword arguments to pass to matplotlib.pyplot.legend()
            categories: Ordered list-like object of categories to be used for categorical plot.
            aspect: ‘auto’, ‘equal’, None or float (default ‘auto’)
            autolim: Update axes data limits to contain the new geometries.
        Returns: axes
        """
        if self._data.empty:
            return
        if boundary:
            data = self._data.copy()
            gtypes = (data.geometry.geom_type == 'Polygon') | (data.geometry.geom_type == 'MultiPolygon')
            data.geometry[gtypes] = data.geometry[gtypes].boundary
        else:
            data = self._data

        if not data.is_empty.all():
            return data.plot(column=column, kind=kind, cmap=cmap, color=color, ax=ax, cax=cax,
                             categorical=categorical, legend=legend, vmin=vmin, vmax=vmax, markersize=markersize,
                             figsize=figsize, legend_kwds=legend_kwds, categories=categories, aspect=aspect,
                             **kwargs)

    # PROPERTIES -------------------------------------------------------------------------------------------------------
    @property
    def properties_tags(self) -> set:
        """Returns set of column names excluding geometry (properties names)"""
        return set(self._data.columns.values) - {GEOMETRY}

    @property
    def properties(self) -> dict:
        """Returns columns excluding geometry (properties) as dict"""
        props = self._data[list(self.properties_tags)].to_dict()
        for k in props.keys():
            props[k] = list(props[k].values())
            if len(props[k]) == 1:
                props[k] = props[k][0]
        return props

    @property
    def geometry(self) -> gpd.GeoSeries:
        """Returns geometry column as a GeoSeries"""
        if self._data.empty:
            return gpd.GeoSeries()
        return self._data.geometry

    @geometry.setter
    def geometry(self, value: Union[gpd.GeoSeries, Sequence[BaseGeometry]]):
        self._data['geometry'] = value

    @property
    def crs(self) -> Optional[CRS]:
        """Returns CRS object or None, if FeatureCollection is empty"""
        try:
            return self._data.crs
        except AttributeError:
            return None

    @property
    def sindex(self):
        """Returns rtree, do not confuse with index, Raises AttributeError if empty"""
        return self._data.sindex

    @property
    def index(self) -> pd.RangeIndex:
        """Returns rows indices, do not confuse with sindex"""
        return self._data.index

    @property
    def columns(self) -> pd.Index:
        """Returns columns as Index object"""
        return self._data.columns

    @property
    def total_bounds(self) -> Tuple[float, float, float, float]:
        """Returns a tuple containing minx, miny, maxx, maxy values for the bounds of the series as a whole.
        Raises AttributeError if empty"""
        return self._data.geometry.total_bounds

    @property
    def dtypes(self):
        """This returns a Series with the data type of each column. """
        return self._data.dtypes

    # CRS TRANSFORMATIONS ----------------------------------------------------------------------------------------------
    def to_crs(self, dst_crs: Union[CRS, str], inplace: bool = False) -> Optional[Self]:
        if self._data.empty:  # empty case first
            if inplace:
                warnings.warn('Attempting to reproject empty FeatureCollection, nothing happened')
                return
            else:
                warnings.warn('Attempting to reproject empty FeatureCollection, returning same empty FeatureCollection')
                return FeatureCollection()

        if isinstance(dst_crs, str) and dst_crs.lower() == 'utm':
            dst_crs = self._data.estimate_utm_crs()
        if inplace:
            self._data.to_crs(dst_crs, inplace=True)
        else:
            return FeatureCollection(self._data.to_crs(dst_crs, inplace=False))

    def to_utm(self, inplace: bool = False) -> Optional[Self]:
        dst_crs = 'utm'
        if inplace:
            self.to_crs(dst_crs, inplace=True)
        else:
            return self.to_crs(dst_crs, inplace=False)

    def estimate_utm_crs(self) -> Optional[CRS]:
        if self._data.empty:
            warnings.warn('Attempting to estimate UTM for empty FeatureCollection, returning None')
            return None
        return self._data.estimate_utm_crs()

    # DATAFRAME OPERATIONS ---------------------------------------------------------------------------------------------
    def groupby(self,
                by: Any = None,
                dropna: bool = True) -> Dict[Hashable, Self]:
        """Group DataFrame using a mapper or by a Series of columns.
        Args:
            by: mapping, function, label, or list of labels, Used to determine the groups for the groupby.
            dropna: If True, and if group keys contain NA values, NA values together with row/column will be dropped.
        Returns dict[group_value, FeatureCollection]"""
        if self._data.empty:
            return dict()
        return {k: FeatureCollection(group) for k, group in self._data.groupby(by, dropna=dropna)}

    def groupby_indexes(self,
                        by: Any = None,
                        dropna: bool = True) -> Dict[Hashable, Sequence[int]]:
        """Group DataFrame using a mapper or by a Series of columns.
        Args:
            by: mapping, function, label, or list of labels, Used to determine the groups for the groupby.
            dropna: If True, and if group keys contain NA values, NA values together with row/column will be dropped.
        Returns dict[group_value, sequence of indexes]"""
        if self._data.empty:
            return dict()
        return self._data.groupby(by, dropna=dropna).indices

    def drop_duplicates(self, subset: Union[str, Sequence[str]], inplace: bool = False) -> Optional[Self]:
        """Return DataFrame with duplicate rows removed.
        Args:
            subset: Only consider certain columns for identifying duplicates, by default use all the columns.
            inplace: inplace
        """
        if self._data.empty:
            return self if not inplace else None
        if inplace:
            self._data.drop_duplicates(subset=subset, ignore_index=True, inplace=inplace)
        else:
            return FeatureCollection(self._data.drop_duplicates(subset=subset, ignore_index=True, inplace=inplace))

    def merge(self,
              right: Union[Self, gpd.GeoDataFrame, pd.DataFrame],
              how: str = 'inner',
              on: Optional[str] = None,
              left_on: Optional[str] = None,
              right_on: Optional[str] = None,
              left_index: bool = False,
              right_index: bool = False,
              sort: bool = False,
              suffixes: Tuple[str, str] = ('_left', '_right'),
              indicator: Union[bool, str] = False,
              validate: Optional[str] = None) -> Self:
        """Merge two GeoDataFrame objects with a database-style join.
        Args:
            right: object to merge with.
            how: {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}, default ‘inner’, Type of merge to be performed.
            on: Column or index level names to join on. These must be found in both DataFrames.
                If on is None and not merging on indexes then this defaults to the intersection of the columns in both
                DataFrames.
            left_on: Column or index level names to join on in the left DataFrame.
            right_on: Column or index level names to join on in the right DataFrame.
            left_index: Use the index from the left DataFrame as the join key(s).
            right_index: Use the index from the right DataFrame as the join key. Same caveats as left_index.
            sort: Sort the join keys lexicographically in the result DataFrame. If False, the order of the join keys
                  depends on the join type (how keyword).
            suffixes: A length-2 sequence where each element is optionally a string indicating the suffix to add
                      to overlapping column names in left and right respectively.
            indicator: If True, adds a column to the output DataFrame called “_merge” with information on the source
                       of each row. The column can be given a different name by providing a string argument.
                       The column will have a Categorical type with the value of “left_only” for observations whose
                       merge key only appears in the left DataFrame, “right_only” for observations whose merge key
                       only appears in the right DataFrame, and “both” if the observation’s merge key is
                       found in both DataFrames.
            validate: If specified, checks if merge is of specified type.
                      “one_to_one” or “1:1”: check if merge keys are unique in both left and right datasets.
                      “one_to_many” or “1:m”: check if merge keys are unique in left dataset.
                      “many_to_one” or “m:1”: check if merge keys are unique in right dataset.
                      “many_to_many” or “m:m”: allowed, but does not result in checks.
        Returns: FeatureCollection
        """
        if isinstance(right, FeatureCollection):
            right = right._data
        result = self._data.merge(right=right, how=how, on=on, left_on=left_on, right_on=right_on,
                                  left_index=left_index, right_index=right_index, sort=sort,
                                  suffixes=suffixes, indicator=indicator, validate=validate)
        result.rename(columns={GEOMETRY+suffixes[0]: GEOMETRY}, inplace=True)
        result = gpd.GeoDataFrame(result, geometry=GEOMETRY, crs=self.crs)
        return FeatureCollection(result)

    # GEODATAFRAME OPERATIONS ------------------------------------------------------------------------------------------
    def explode(self, inplace: bool = False) -> Optional[Self]:
        """Split GeometryCollections into simple Geometries"""
        if self._data.empty:
            return self if not inplace else None
        if inplace:
            self._data = self._data.explode(ignore_index=True, index_parts=False)
        else:
            return FeatureCollection(self._data.explode(ignore_index=True, index_parts=False))

    def dissolve(self, by=None, aggfunc='first', as_index=False, sort=False,
                 observed=False, dropna=True, inplace=False) -> Optional[Self]:
        """Dissolve geometries within groupby into single observation.
        Args:
            by: Column whose values define groups to be dissolved.
                If None, whole GeoDataFrame is considered a single group.
            aggfunc: aggregation function for manipulation of data associated with each group.
            as_index: If true, groupby columns become index of result.
            sort: Sort group keys. Get better performance by turning this off.
            observed: This only applies if any of the groupers are Categoricals.
                      If True: only show observed values for categorical groupers.
                      If False: show all values for categorical groupers.
            dropna: if group keys contain NA values, NA values together with row/column will be dropped.
            inplace: inplace
        """
        if self._data.empty:
            return self if not inplace else None
        if inplace:
            self._data = self._data.dissolve(by=by, aggfunc=aggfunc, as_index=as_index, sort=sort,
                                             observed=observed, dropna=dropna)
        else:
            return FeatureCollection(self._data.dissolve(by=by, aggfunc=aggfunc, as_index=as_index, sort=sort,
                                                         observed=observed, dropna=dropna))

    def sjoin(self,
              right: Union[Self, gpd.GeoDataFrame],
              how: str = 'inner',
              predicate: str = 'intersects',
              lsuffix: str = 'left',
              rsuffix: str = 'right') -> Self:
        """Spatial join of two GeoDataFrames.
        Args:
            right: other FeatureCollection
            how: The type of join, ‘left’, 'right’, ‘inner’
            predicate: Binary predicate. Valid values are determined by the spatial index used.
            lsuffix: Suffix to apply to overlapping column names (left GeoDataFrame).
            rsuffix: Suffix to apply to overlapping column names (right GeoDataFrame).
        Returns: FeatureCollection
        """
        if isinstance(right, FeatureCollection):
            right = right._data

        if self._data.empty and right.empty:  # both empty
            return self

        if right.empty:   # TODO: test this
            if how == 'outer':
                return self
            else:
                return FeatureCollection()

        if self._data.empty:   # TODO: test this
            if how != 'inner':
                return self
            else:
                return FeatureCollection()

        else:  # both are OK
            return FeatureCollection(self._data.sjoin(right, how=how, predicate=predicate, lsuffix=lsuffix,
                                                      rsuffix=rsuffix))

    def sjoin_nearest(self,
                      right: Union[Self, gpd.GeoDataFrame],
                      how: str = 'inner',
                      max_distance: Optional[float] = None,
                      lsuffix: str = 'left',
                      rsuffix: str = 'right',
                      distance_col: Optional[str] = None) -> Self:
        """Spatial join of two GeoDataFrames based on the distance between their geometries.
        Results will include multiple output records for a single input record where there are multiple
        equidistant nearest or intersected neighbors.
        Args:
            right: other FeatureCollection
            how: The type of join, ‘left’, 'right’, ‘inner’
            max_distance: Maximum distance within which to query for nearest geometry.
            lsuffix: Suffix to apply to overlapping column names (left GeoDataFrame).
            rsuffix: Suffix to apply to overlapping column names (right GeoDataFrame).
            distance_col: save the distances computed between matching geometries under a column of this name
        Returns: FeatureCollection
        """
        if isinstance(right, FeatureCollection):
            right = right._data

        if self._data.empty and right.empty:  # both empty
            return self

        if right.empty:  # TODO: test this
            if how == 'outer':
                return self
            else:
                return FeatureCollection()

        if self._data.empty:  # TODO: test this
            if how != 'inner':
                return self
            else:
                return FeatureCollection()

        else:  # both are OK
            return FeatureCollection(self._data.sjoin_nearest(right, how=how, max_distance=max_distance,
                                                              lsuffix=lsuffix, rsuffix=rsuffix,
                                                              distance_col=distance_col))

    # GEOMETRY TRANSFORMATIONS -----------------------------------------------------------------------------------------

    def area(self) -> pd.Series:
        if self._data.empty:
            return pd.Series(dtype=float)
        return self._data.geometry.area

    def length(self) -> pd.Series:
        if self._data.empty:
            return pd.Series(dtype=float)
        return self._data.geometry.length

    def distance(self, other: Union[BaseGeometry, pd.Series, Sequence[BaseGeometry]]) -> pd.Series:
        if self._data.empty:
            return pd.Series(dtype=float)
        return self._data.geometry.distance(other, align=False)

    def hausdorff_distance(self, other: Union[BaseGeometry, pd.Series, Sequence[BaseGeometry]]) -> pd.Series:
        if self._data.empty:
            return pd.Series(dtype=float)
        return self._data.geometry.hausdorff_distance(other, align=False)

    def buffer(self, distance: float, resolution: float = 16, cap_style: str = 'round',
               join_style: str = 'round', mitre_limit: float = 5.0, single_sided: bool = False,
               inplace: bool = False) -> Optional[Self]:
        """Buffer
        Args:
            distance: The radius of the buffer in the Minkowski sum (or difference).
                      If array or Series are used then it must have same length as the GeoSeries.
            resolution: The resolution of the buffer around each vertex.
            cap_style: {‘round’, ‘square’, ‘flat’}, default ‘round’
                       Specifies the shape of buffered line endings. 'round' results in circular line endings
                       (see resolution). Both 'square' and 'flat' result in rectangular line endings,
                        'flat' will end at the original vertex, while 'square' involves adding the buffer width.

            join_style: {‘round’, ‘mitre’, ‘bevel’}, default ‘round’ Specifies the shape of buffered line midpoints.
                        'round' results in rounded shapes. 'bevel' results in a beveled edge that touches the
                         original vertex. 'mitre' results in a single vertex that is beveled depending on the
                         mitre_limit parameter.

            mitre_limit: Crops of 'mitre'-style joins if the point is displaced from the buffered vertex by more
                         than this limit.
            single_sided: Only buffer at one side of the geometry.
            inplace: inplace
        """
        if self._data.empty:
            return self if not inplace else None
        if inplace:
            self._data.geometry = self._data.geometry.buffer(distance=distance, resolution=resolution,
                                                             cap_style=cap_style, join_style=join_style,
                                                             mitre_limit=mitre_limit, single_sided=single_sided)
        else:
            data = self._data.copy()
            data.geometry = data.geometry.geometry.buffer(distance=distance, resolution=resolution, cap_style=cap_style,
                                                          join_style=join_style, mitre_limit=mitre_limit,
                                                          single_sided=single_sided)
            return FeatureCollection(data)

    def simplify(self, tolerance: float, preserve_topology: bool = True, inplace: bool = False) -> Optional[Self]:
        """ Simplify
        Args:

        """
        if self._data.empty:
            return self if not inplace else None
        if inplace:
            self._data.geometry = self._data.geometry.simplify(tolerance=tolerance, preserve_topology=preserve_topology)
        else:
            data = self._data.copy()
            data.geometry = data.geometry.simplify(tolerance=tolerance, preserve_topology=preserve_topology)
            return FeatureCollection(data)

    def merge_connected_geometries(self, buffer: float = 0., inplace: bool = False) -> Optional[gpd.GeoSeries]:
        # TODO: figure out how to implement inplace (the problem is that the result will have different length)
        if self._data.empty:
            return gpd.GeoSeries() if not inplace else None
        if inplace:
            raise NotImplementedError
        merged = self._data.geometry.buffer(buffer).union_all().buffer(-buffer)
        return gpd.GeoSeries([p for p in merged.geoms] if isinstance(merged, shapely.MultiPolygon) else merged,
                             crs=self.crs)

    def make_valid(self, dropna: bool = True, drop_empty: bool = True,
                   explode: bool = True, remove_repeated_points: bool = True,
                   inplace: bool = False) -> Optional[Self]:
        """Make geometries valid with shapely.
        Args:
            dropna - rop NaN or None values
            drop_empty - drop empty geometries
            explode - explode GeometryCollections
            remove_repeated_points - remove repeated points with 0 tolerance
            inplace - inplace
        Returns:
             FeatureCollection if not inplace"""
        if self.empty:
            return self if not inplace else None
        if inplace:
            self._data.geometry = self._data.geometry.make_valid()
            if explode:
                self._data = self._data.explode(ignore_index=True, index_parts=False)
            if dropna:
                self._data.dropna(axis=0, subset='geometry', inplace=True)
            if drop_empty:
                self._data = self._data[~self._data.is_empty]
            self._data.reset_index(drop=True, inplace=True)
            if remove_repeated_points:
                self._data.geometry = gpd.GeoSeries(shapely.remove_repeated_points(self._data.geometry.values),
                                                    crs=self._data.crs)
            self._data.geometry = self._data.geometry.make_valid()  # TODO: avoid double make valid
        else:
            data = self._data.copy()
            data.geometry = data.geometry.make_valid()
            if explode:
                data = data.explode(ignore_index=True, index_parts=False)
            if dropna:
                data.dropna(axis=0, subset='geometry', inplace=True)
            if drop_empty:
                data = data[~data.is_empty]
            data.reset_index(drop=True, inplace=True)
            if remove_repeated_points:
                data.geometry = gpd.GeoSeries(shapely.remove_repeated_points(data.geometry.values),
                                              crs=data.crs)
            data.geometry = data.geometry.make_valid()  # TODO: avoid double make valid
            return FeatureCollection(data)

    def remove_repeated_points(self, tolerance: float = 0, inplace: bool = False) -> Optional[Self]:
        """From the start of the coordinate sequence, each next point within the tolerance is removed.
        Args:
            tolerance - Remove all points within this distance of each other.
                        Use 0.0 to remove only exactly repeated points (the default).
            inplace - inplace
        Returns:
            FeatureCollection in not inplace"""
        if self.empty:
            return self if not inplace else None
        if inplace:
            self._data.geometry = gpd.GeoSeries(shapely.remove_repeated_points(self._data.geometry.values,
                                                                               tolerance=tolerance),
                                                crs=self._data.crs)
        else:
            data = self._data.copy()
            data.geometry = gpd.GeoSeries(shapely.remove_repeated_points(data.geometry.values,
                                                                         tolerance=tolerance),
                                          crs=data.crs)
            return FeatureCollection(data)


def concatenate(fcs: Sequence[FeatureCollection]) -> FeatureCollection:
    return FeatureCollection(pd.concat([f._data for f in fcs], ignore_index=True))


def show_fcs(*fcs: FeatureCollection, **kwargs):
    from matplotlib import pyplot as plt
    colors = ['blue', 'orange', 'lime', 'red', 'green', 'cyan', 'yellow']
    _, ax = plt.subplots()
    for i, fc in enumerate(fcs):
        fc.plot(ax=ax, color=colors[i % len(colors)], alpha=max(0.3, 1 / len(fcs)), **kwargs)
    plt.show()

def query_to_dict(query_res: np.array) -> dict:
    """Convert FeatureCollection.query() result to dict like {feature_idx: [intersecting indexes, ]}"""
    assert query_res.ndim == 2 and query_res.shape[0] == 2
    dict_res = dict()
    for idx in set(list(query_res[1])):
        dict_res[idx] = list(query_res[0, query_res[1]==idx])
    return dict_res
