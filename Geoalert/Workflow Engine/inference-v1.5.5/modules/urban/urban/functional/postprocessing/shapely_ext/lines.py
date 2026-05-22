import shapely
import shapely.geometry


def extend_line(p1: shapely.geometry.Point, p2: shapely.geometry.Point, extension: float = 1.):
    """Extend line for `extension / 2` in each direction"""
    
    x1, y1 = list(p1.coords)[0]
    x2, y2 = list(p2.coords)[0]
    
    init_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    k = (extension / 2) / init_length
    
    x2e = x2 + (x2 - x1) * k
    y2e = y2 + (y2 - y1) * k
    
    x1e = x1 - (x2 - x1) * k
    y1e = y1 - (y2 - y1) * k
    
    return (x1e, y1e), (x2e, y2e)
