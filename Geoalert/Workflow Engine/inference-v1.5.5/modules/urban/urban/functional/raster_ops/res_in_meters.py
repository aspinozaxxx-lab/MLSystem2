def get_resolution_in_meters(crs, res):
    """Get resolution in meters
        for EPSG:3857 resolution is in metre => factor = 1
        for EPSG:6360 resolution is in foot => factor = 0.3048... (foot/metre)
        for EPSG:4326 resolution is in degree => factor = 0.01745...(radian/degree) but we need in metre => factor = 111134 (metre/degree)
    """
    units, factor = getattr(crs, 'units_factor', (None, 1))

    if units == 'degree':
        factor = 111134

    resolution = res * factor

    return resolution
