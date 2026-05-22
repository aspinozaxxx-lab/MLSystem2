import os
from pymongo import MongoClient
from gpdadapter import FeatureCollection
from shapely.geometry import Polygon, mapping
import geopandas as gpd


class MongoAPI:

    def __init__(self, db, collection, uri=None):

        if uri is None:
            uri = os.environ.get('MONGODB_URI')

        self.client = MongoClient(uri)
        self.db = getattr(self.client, db)
        self.collection = getattr(self.db, collection)

    def query(self, data):  # TODO: staticmethod?
        return {
            'geometry': {
                '$geoIntersects': {
                    '$geometry': data
                }
            }
        }

    def _convert_to_fc(self, data):  # TODO: staticmethod?
        fc = FeatureCollection(gpd.GeoDataFrame.from_features(data))
        return fc

    def get(self, geometry: Polygon):
        query = self.query(mapping(geometry))
        data = self.collection.find(query)
        fc = self._convert_to_fc(data)
        return fc
