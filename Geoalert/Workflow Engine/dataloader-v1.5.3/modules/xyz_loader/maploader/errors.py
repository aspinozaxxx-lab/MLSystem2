class MaploaderError(Exception):
    def __init__(self, message, **parameters):
        self.error_code = self.__class__.__name__
        self.parameters = parameters
        try:
            self.message = message.format(**parameters)
        except KeyError:
            # if there are not enough parameters to format the string -
            # it means that there is error in error class definition/call
            raise AssertionError(f"Error classes must provide all keys for message formatting. "
                                 f"The class {self.error_code} with message \"{message}\" "
                                 f"has incompatible parameters: {parameters}")
        super().__init__(self.message)

    def asdict(self):
        return {"code": self.error_code,
                "parameters": {key: str(value) for key, value in self.parameters.items()},
                "message": self.message}

    def __str__(self):
        return f"{self.error_code}: {self.message}"


class WrongNdim(MaploaderError):
    def __init__(self, real_ndim):
        super().__init__(message="Tile must be 3-dimensional array, but has {real_ndim} dimensions",
                         real_ndim=real_ndim)


class WrongChannelsNum(MaploaderError):
    def __init__(self, expected_nchannels, real_nchannels):
        super().__init__(message="Image must have {expected_nchannels} channels, but has {real_nchannels} channels",
                         expected_nchannels=expected_nchannels,
                         real_nchannels=real_nchannels)


class WrongTileSize(MaploaderError):
    def __init__(self, expected_size, real_size):
        super().__init__(message="Tile must have size {expected_size}, but has {real_size}",
                         expected_size=expected_size,
                         real_size=real_size)


class WrongSourceType(MaploaderError):
    def __init__(self, source_type):
        super().__init__(message="Allowed source types are: xyz, tms, quadkey. Got {source_type}",
                         source_type=source_type)


class TileNotLoaded(MaploaderError):
    def __init__(self, tile_location, proxy, exception_message, http_status):
        super().__init__(message="Error downloading tile at {tile_location}: tile server replied with {status}. "
                                 "Message: {exception_message}" + f"Full tile location: {tile_location}, proxy: {proxy}",
                         tile_location=tile_location.split('?')[0],
                         exception_message=exception_message,
                         status=http_status)


class TileNotReadable(MaploaderError):
    def __init__(self, tile_location, proxy, exception_message):
        super().__init__(message="Response content at {tile_location} cannot be decoded as an image."
                                 "Message: {exception_message}" + f"Full tile location: {tile_location}, proxy: {proxy}",
                         tile_location=tile_location.split('?')[0],
                         exception_message=exception_message)


class CrsIsNotSupported(MaploaderError):
    def __init__(self, supported_projections, real_projection):
        super().__init__(message="Projection {real_projection} is not supported, use one of {supported_projections}.",
                         supported_projections=supported_projections,
                         real_projection=real_projection)


class MaploaderInternalError(MaploaderError):
    def __init__(self, error_message):
        super().__init__(message="Internal error in maploader: {error_message}",
                         error_message=error_message)