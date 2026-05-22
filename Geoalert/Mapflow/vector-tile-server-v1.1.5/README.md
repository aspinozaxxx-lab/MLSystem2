# API
## TileJson layer description
Returns the layer description according to [tilejson spec](https://github.com/mapbox/tilejson-spec).  
`GET /api/layers/{layerId}.json`

## MVT tile
Returns an MVT tile selected by `X`, `Y`, `Z` coordinates in the `WebMercator` tile pyramid.  
`GET /api/layers/{layerId}/tiles/{z}/{x}/{y}.vector.pbf`

## API query parameters
Both Tile and TileJson API methods support the following optional HTTP query parameters:  
Example: `GET /api/layers/8fd2c319-6a6f-42cc-8c29-c8263114a04c.json?simplify_factor=0.001&max_features=1000`  

| Name                 | Description                                                                                                   | Default value              |
| -------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------- |
| min_area_factor      | A factor used for filtering features when zoom <= `simplify_max_zoom`. Higher value means fewer features.     | $DEFAULT_MIN_AREA_FACTOR   |
| simplify_factor      | A factor used for geometry simplification when zoom <= `simplify_max_zoom`                                    | $DEFAULT_SIMPLIFY_FACTOR   |
| simplify_max_zoom    | Simplification and filtering are performed when zoom <= `simplify_max_zoom`                                   | $DEFAULT_SIMPLIFY_MAX_ZOOM |
| min_zoom             | An empty tile will be returned when zoom < `min_zoom`                                                         | $DEFAULT_MIN_ZOOM          |
| max_features         | At most `max_features` features will be rendered per tile                                                     | $DEFAULT_MAX_FEATURES      |
| points               | If set to `true`, geometries will be converted to points                                                      | false                      |

# Env variables

## Default values for API parameters 
| Name                         | Description                                                                                                   | Default value    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------- |
| DEFAULT_MIN_AREA_FACTOR      | A factor used for filtering features when zoom <= `simplify_max_zoom`. Higher value means fewer features.     | 0.0000001        |
| DEFAULT_SIMPLIFY_FACTOR      | A factor used for geometry simplification when zoom <= `simplify_max_zoom`                                    | 0.0006           |
| DEFAULT_SIMPLIFY_MAX_ZOOM    | Simplification and filtering are performed when zoom <= `simplify_max_zoom`                                   | 17               |
| DEFAULT_MIN_ZOOM             | An empty tile will be returned when zoom < `min_zoom`                                                         | 14               |
| DEFAULT_MAX_FEATURES         | At most `max_features` features will be rendered per tile                                                     | 10000            |

## DB parameters 
| Name              | Description       | Default value   |
|-------------------|-------------------|-----------------|
| DATABASE_PORT     | Database port     | 5432            |
| DATABASE_HOST     | Database host     | vector-database |
| DATABASE_NAME     | Database name     | vector_db       |
| DATABASE_USER     | Database user     | postgres        |
| DATABASE_PASSWORD | Database password | 1234Qq          |
| DATABASE_SCHEMA   | Database schema   | public          |

## Misc parameters 
| Name                 | Description                                           | Default value          |
| -------------------- | ----------------------------------------------------- | ---------------------- |
| EXTERNAL_URL         | URL to this app, accessible from the internet         | http://localhost:8600  |
