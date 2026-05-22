from gpdadapter import FeatureCollection, concatenate, DEFAULT_CRS
import pandas as pd
import geopandas as gpd
import pytest
import numpy as np
import shapely


def test_create_empty_fc():
    fc = FeatureCollection()  # from None
    assert fc.empty and len(fc) == 0
    assert len(fc.columns) == 1 and fc.columns[0] == 'geometry'
    assert fc.crs is None

    fc = FeatureCollection([])  # from list
    assert fc.empty and len(fc) == 0
    assert len(fc.columns) == 1 and fc.columns[0] == 'geometry'
    assert fc.crs is None

    fc = FeatureCollection(dict())  # from dict
    assert fc.empty and len(fc) == 0
    assert len(fc.columns) == 1 and fc.columns[0] == 'geometry'
    assert fc.crs is None

    fc = FeatureCollection(tuple())
    assert fc.empty and len(fc) == 0
    assert len(fc.columns) == 1 and fc.columns[0] == 'geometry'
    assert fc.crs is None

    fc = FeatureCollection({'geometry': []})
    assert fc.empty and len(fc) == 0
    assert len(fc.columns) == 1 and fc.columns[0] == 'geometry'
    assert fc.crs is None

    # TODO:
    # fc = FeatureCollection([{'geometry': []}])
    # assert fc.empty and len(fc) == 0
    # assert len(fc.columns) == 1 and fc.columns[0] == 'geometry'
    # assert fc.crs is None

    fc = FeatureCollection(pd.Series())
    assert fc.empty and len(fc) == 0
    assert len(fc.columns) == 1 and fc.columns[0] == 'geometry'
    assert fc.crs is None

    fc = FeatureCollection(gpd.GeoSeries())
    assert fc.empty and len(fc) == 0
    assert len(fc.columns) == 1 and fc.columns[0] == 'geometry'
    assert fc.crs is None

    fc = FeatureCollection(gpd.GeoSeries([shapely.Polygon([[0, 0], [0, 1], [1, 1]])], crs='EPSG:3857'))
    assert len(fc) == 1
    assert len(fc.columns) == 1 and fc.columns[0] == 'geometry'
    assert fc.crs == 'EPSG:3857'

    fc = FeatureCollection(pd.DataFrame())
    assert fc.empty and len(fc) == 0
    assert len(fc.columns) == 1 and fc.columns[0] == 'geometry'
    assert fc.crs is None

    fc = FeatureCollection(gpd.GeoDataFrame())
    assert fc.empty and len(fc) == 0
    assert len(fc.columns) == 1 and fc.columns[0] == 'geometry'
    assert fc.crs is None


def test_create_empty_fc_with_additional_columns():
    fc = FeatureCollection({'geometry': [], 'asdf': []})
    assert fc.empty and len(fc) == 0
    assert len(fc.columns) == 2 and fc.columns[0] == 'geometry' and fc.columns[1] == 'asdf'

    with pytest.raises(KeyError):  # todo: raise ValueError instead
        fc = FeatureCollection({'asdf': []})  # should have 'geometry' column


def test_create_non_empty_fc():
    fc = FeatureCollection({'foo': [1, np.nan, 3],
                            'bar': [4, np.nan, None],
                            'geometry': [shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1]]),  # valid
                                         shapely.Polygon([[0, 0], [1, 0], [0, 1], [1, 1]]),  # invalid
                                         shapely.Polygon([])]})  # empty
    assert len(fc) == 3 and len(fc.columns) == 3

    fc = FeatureCollection(gpd.GeoDataFrame({'foo': [1, 2, 3], 'bar': [4, 5, 6], 'geometry': [shapely.Polygon([])] * 3},
                            crs='epsg:4326'))
    assert len(fc) == 3 and len(fc.columns) == 3

    fc = FeatureCollection(shapely.Point())
    assert len(fc) == 1 and len(fc.columns) == 1 and fc.crs == DEFAULT_CRS

    fc = FeatureCollection([shapely.Polygon([])] * 3, crs='epsg:3857')
    assert len(fc) == 3 and len(fc.columns) == 1 and fc.crs == 'epsg:3857'

    fc = FeatureCollection(gpd.GeoSeries([shapely.Polygon([])] * 3, crs='epsg:3857'))
    assert len(fc) == 3 and len(fc.columns) == 1 and fc.crs == 'epsg:3857'

    fc = FeatureCollection([pd.Series({'foo': 1, 'bar': 2, 'geometry': shapely.Polygon([])})] * 3)
    assert len(fc) == 3 and len(fc.columns) == 3 and fc.crs == DEFAULT_CRS

    with pytest.raises(ValueError):
        fc = FeatureCollection([pd.Series({'foo': 1, 'bar': 2})] * 3)  # no geometry column

    fc = FeatureCollection([FeatureCollection(
        {'foo': 1, 'bar': 2, 'geometry': shapely.Polygon([])}, crs='epsg:3857')] * 3)
    assert len(fc) == 3 and len(fc.columns) == 3 and fc.crs =='epsg:3857'


def test_getitem():
    fc = FeatureCollection({'foo': [1, 2, 3], 'bar': [4, 0, 6], 'geometry': [shapely.Polygon([])] * 3})

    value = fc[0, 'foo']
    assert value == 1

    value = fc[0, 'geometry']
    assert value == shapely.Polygon([])

    value = fc[0]
    assert isinstance(value, FeatureCollection) and len(value) == 1

    value = fc[:, 'foo']
    assert isinstance(value, pd.Series) and len(value) == 3

    value = fc[:, 'geometry']
    assert isinstance(value, FeatureCollection) and len(value) == 3

    value = fc.geometry
    assert isinstance(value, gpd.GeoSeries) and len(value) == 3

    value = fc[0, ('foo', 'bar')]
    assert isinstance(value, pd.Series)

    value = fc[:, ('foo', 'bar')]
    assert isinstance(value, pd.DataFrame)

    value = fc[:2, ('foo', 'geometry')]
    assert isinstance(value, FeatureCollection)

    value = fc[[0, 2]]
    assert isinstance(value, FeatureCollection) and len(value) == 2

    value = fc[fc[:, 'foo'] > 1]
    assert isinstance(value, FeatureCollection) and len(value) == 2

    value = fc[(fc[:, 'foo'] + fc[:, 'bar'] > 5) & (fc[:, 'bar'] < 7)]
    assert isinstance(value, FeatureCollection) and len(value) == 1


def test_setitem():
    fc = FeatureCollection({'foo': [1, 2, 3], 'bar': [4, 5, 6], 'geometry': [shapely.Polygon([])] * 3})
    for f in fc:
        f[:, 'foo'] = 1000
    assert (fc[:, 'foo'] == pd.Series([1, 2, 3])).all()

    fc[0, 'foo'] = 10
    assert fc[0, 'foo'] == 10

    fc[0, 'buzz'] = 10
    #assert (fc[:, 'buzz'] == pd.Series([10, None, None])).all() TODO: fix
    assert fc[0, 'buzz'] == 10 and len(fc.columns)==4

    fc[0] = {'foo': 0, 'bar': 1, 'geometry': shapely.Point(), 'buzz': 30}
    assert fc[0, 'foo'] == 0

    fc[0] = {'foo': 10, 'geometry': shapely.Polygon([])}
    assert fc[0, 'foo'] == 10 # and fc[0, 'bar'] == np.nan TODO: fix

    fc[0] = FeatureCollection({'foo': [0], 'bar': [1], 'geometry': [shapely.Point()], 'buzz': [40]})
    assert fc[0, 'buzz'] == 40

    fc[:, 'foo'] = 100
    assert (fc[:, 'foo'] == 100).all()

    fc[:2, 'foo'] = 200  # slice as first index
    assert (fc[:2, 'foo'] == 200).all()

    fc[[0, 2], 'foo'] = 400  # sequence as first index
    assert (fc[[0, 2], 'foo'] == 400).all()

    fc[np.array([0, 2]), 'foo'] = 400  # sequence as first index
    assert (fc[[0, 2], 'foo'] == 400).all()

    fc[:, 'bar'] = [1, 2, 3]
    assert (fc[:, 'bar'] == pd.Series([1, 2, 3])).all()

    fc[:, 'objects'] = None
    assert 'objects' in fc.columns

    fc[:, 'ints'] = pd.Series(dtype='Int32')
    assert 'ints' in fc.columns

    fc[:, 'ints'] = pd.Series(dtype='Int32')
    assert 'ints' in fc.columns

    fc[:, 'ints2'] = np.arange(0, 3)
    assert (fc[:, 'ints2'] == pd.Series([0, 1, 2])).all()

    fc[:, 'ni'] = fc[:, 'foo'] + fc[:, 'bar']
    assert (fc[:, 'ni'] == fc[:, 'foo'] + fc[:, 'bar']).all()

    fc.drop('buzz', axis=1, inplace=True)
    assert 'buzz' not in fc.columns

    fc.append({'bar': 0, 'geometry': shapely.Polygon([]), 'buzz': 'asdf'})
    assert len(fc) == 4

    fc.drop(1, inplace=True)
    assert len(fc) == 3

    fc[:, 'strings'] = 'string'
    assert all(fc[i, 'strings'] == 'string' for i in range(len(fc)))

    fc[:2, 'strings'] = ['string1', 'string2']
    assert fc[1, 'strings'] == 'string2'

    fc[:2, 'non_existing_column'] = ['a', 'b']  # TODO: shouldn't this raise an exception?
    assert fc[1, 'non_existing_column'] == 'b'

    fc[:, 'strings'] = 'string3'
    assert fc[1, 'strings'] == 'string3'