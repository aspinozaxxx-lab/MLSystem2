import shapely
from ..base import Brick
from ..functional import io, postprocessing
from shapely.affinity import affine_transform
from gpdadapter import FeatureCollection
from loguru import logger
import topojson as tp
from pydantic import Field


class VectorizeAsLines(Brick):
    
    """
    Makes a polygon map of linear objects (roads etc.) from raster mask
    - the skeleton is transformed to graph via sknw
    - the graph lines are simplified and turned to the shapely lines
    - the lines are buffered to get polygons compatible with FeatureCollections
    Width is determined from the real mask width at each segment in the input mask.

    Args:
        mask_raster (str): Band name of input segmentation mask
        skeleton_raster (str): Band name of input skeleton of segmentation mask
        out_vector (str): name of the output vector file
        min_width (float): minimum allowed width for the road (in Bands' projection units)
        max_width (float): maximum allowed width for the road (in Bands' projection units)
        approx (float): parameter for cv2.approxPolyDP line simplification
        min_terminal_length (float): the terminal lines with less length will be deleted from graph
        merge (bool): if True, all the line segments are merged, which produces a single Polygon from each connected
                      component of the objects else, every line segment from a joint to another will be
                      represented by a separate polygon
    """
    mask_raster: str
    skeleton_raster: str
    out_vector: str
    polygonize: bool = Field(True)
    min_width: float = Field(5)
    max_width: float = Field(5)
    approx: float = Field(2)
    min_terminal_length: float = Field(10)
    merge: bool = Field(False)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.min_width > self.max_width:
            raise ValueError('Max width must greater or equal to min width')

    def __call__(self, path):
        bc = io.read_bc(path, [self.mask_raster, self.skeleton_raster])
        mask = bc[0]
        skeleton = bc[1]

        linestrings = postprocessing.network.skelet2linestrings(skeleton, self.approx, self.min_terminal_length)
        logger.debug(f"Processing {len(linestrings)} linestrings")
        matrix = [bc.transform[3], -bc.transform[4], -bc.transform[0],
                  bc.transform[1], bc.transform[2],  bc.transform[5]]

        features = [affine_transform(line, matrix) for line in linestrings]
        
        if self.polygonize:
            features = postprocessing.network.linestrings2polygons(features, mask, self.min_width,
                                                                   self.max_width, self.merge)

        fc = FeatureCollection(features, bc.crs)
        
        if self.merge and self.polygonize:
            fc = postprocessing.flatten_multipolygons.flatten_multipolygons(fc)
            fc = postprocessing.flatten_multipolygons.merge_connected_polygons(fc)
        
        io.save_fc(fc, path, self.out_vector)


class SnapLines(Brick):
    
    """
    Snap lines with same endnodes.
    (lines that were broken into pieces after skeletonization)
    
    Args:
        input (str): filename of feature collection 
        output (str): filename of feature collection for output
        angle (float): min difference in lines angles (degrees) to snap
        crs (str): projection 
    """
    input: str
    output: str
    angle: float = Field(40)
    crs: str = Field("utm")
    verbose: bool = Field(True)

    def __call__(self, path):
        fc = io.read_fc(path, self.input, crs=self.crs)
        fc = postprocessing.linestrings.snap_lines(fc, self.angle, self.verbose)
        io.save_fc(fc, path, self.output)

    
class FilterShortLines(Brick):
    
    """
    Filter short lines, this also checks if line connected to other lines
    and does not remove it, if number of connections >= min_connections
    
    Args:
        input (str): filename of feature collection 
        output (str): filename of feature collection for output
        min_length (float) - min length of line to be filtered if projection units
        min_connections (int) - lines, connected to >= lines will not be filtered
        endnodes_only (bool) - if True, connections will be checked only at line endnodes
        crs (str): projection 
    """
    input: str
    output: str
    min_length: float = Field(30)
    min_connections: int = Field(2)
    endnodes_only: bool = Field(False)
    crs: str = Field("utm")

    def __call__(self, path):
        fc = io.read_fc(path, self.input, crs=self.crs)
        filter_ = postprocessing.linestrings.filter_short_lines(
            fc,
            self.min_length,
            self.min_connections,
            self.endnodes_only
        )
        fc = fc.filter(filter_)
        io.save_fc(fc, path, self.output)


class LineStings2Polygons(Brick):
    """Bufferize LineStrings to get polygons.
    Width is determined from the real mask width 
    at each segment in the input mask.

    Args:
        ls_vector (str): filename of feature collection with linestring geometry
        mask_raster (str): Band name of input segmentation mask
        output (str): name of the output vector file
        min_width (float): minimum allowed width for the road (in Bands' projection units)
        max_width (float): maximum allowed width for the road (in Bands' projection units)
        merge (bool): if True, all the line segments are merged, which produces a single Polygon from each connected
                      component of the objects else, every line segment from a joint to another will be
                      represented by a separate polygon
    """
    ls_vector: str
    mask_raster: str
    output: str
    min_width: float = Field(3)
    max_width: float = Field(3)
    merge: bool = Field(False)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.min_width > self.max_width:
            raise ValueError('Max width must greater or equal to min width')

    def __call__(self, path):
        mask = io.read_bc(path, [self.mask_raster])[0]
        fc = io.read_fc(path, self.ls_vector, crs=mask.crs)
        
        polygons = postprocessing.network.linestrings2polygons(
            list(fc.geometry),
            mask,
            self.min_width,
            self.max_width,
            self.merge)
        
        fc = FeatureCollection(polygons, crs=fc.crs)
        
        if self.merge:
            fc = postprocessing.flatten_multipolygons.flatten_multipolygons(fc)
            fc = postprocessing.flatten_multipolygons.merge_connected_polygons(fc)
            
        io.save_fc(fc, path, self.output, explode=True, keep_only_geometry_types=shapely.Polygon)


class RefineCrossroads(Brick):
    """Fix crossroads that were broken into several parts after skeletonization.
    We assume that some crossroads are broken into >—< (two points, in each
    two lines connected and some line at center that should be removed).
    This brick is designed to calculate true center of crossroad and snap lines
    endnodes to this point and remove line at the center.
    Basically, it replace >—< with ><.

    Args:
        input (str): filename of feature collection 
        output (str): filename of feature collection for output
        num_lines (int): number of lines ending at one point for crossroad search
        snap_dist (float): max distance between points of broken crossroad
        merge_dist (float): crossroads at this distance will merge into one
        min_length (float): min line length (for filtering)
        min_connections (int): min lines connected to given (for filtering)
        endnodes_only (bool): search crossroads only at lines endnodes
        crs (str): projection 
    """
    input: str
    output: str
    num_lines: int = Field(3)
    snap_dist: float = Field(5)
    merge_dist: float = Field(10)
    min_length: float = Field(10)
    min_connections: int = Field(2)
    endnodes_only: bool = Field(True)
    crs: str = Field("utm")
    verbose: bool = Field(True)

    def __call__(self, path):
        fc = io.read_fc(path, self.input, crs=self.crs)
        
        fc = postprocessing.linestrings.snap_crossroads(
            fc,
            self.num_lines,
            self.snap_dist,
            self.merge_dist,
            self.min_length,
            self.min_connections,
            self.endnodes_only,
            self.verbose
        )
        
        io.save_fc(fc, path, self.output)


class TopoSimplify(Brick):
    
    """
    Simplify road network remaining topology
    Args:
        input (srt): filename of feature collection 
        output (str): filename of feature collection for output
        tolerance (float): simplification parameter in crs units
    """
    input: str
    output: str
    tolerance: float = Field(4)

    def __call__(self, path):
        fc = io.read_fc(path, self.input)
        tolerance = postprocessing.linestrings.meters_to_degrees(self.tolerance, fc.estimate_utm_crs())
        simplify = tp.Topology(data=fc.to_json_dict(), topology=True)
        fc = FeatureCollection.from_json_str(simplify.toposimplify(tolerance).to_geojson())
        io.save_fc(fc, path, self.output, keep_only_geometry_types=shapely.LineString)


class CloseGaps(Brick):
    """
    Close small gaps in road network.
    We create two buffers (for both endnodes) for each line and check for other
    lines endnodes intersects this buffer and have similar slope.
    
    Args:
        input (str): filename of feature collection 
        output (str): filename of feature collection for output
        angle (float): max difference in lines angles for merging in degrees
        buff_length (float): buffer length in crs units
        buff_width (float): buffer width in crs units
        buff_type (str): buffer type - either rectangle or triangle
        simplify_param (float): tolerance param for merged line simplification
        intersect_buff (float): buffer param for fc to avoid intersecting and
                                              duplicating lines after merging
        crs (str): projection
    """
    input: str
    output: str
    angle: float = Field(10)
    buff_length: float = Field(150)
    buff_width: float = Field(12)
    buff_type: str = Field('rectangle')
    intersection_mode: str = Field('endnodes')
    simplify_param: float = Field(0)
    intersect_buff: float = Field(3)
    crs: str = Field("utm")
    verbose: bool = Field(True)

    def __call__(self, path):
        fc = io.read_fc(path, self.input, crs=self.crs)
        
        fc = postprocessing.linestrings.close_gaps(
            fc,
            self.angle,
            self.buff_length,
            self.buff_width,
            self.buff_type,
            self.intersection_mode,
            self.simplify_param,
            self.intersect_buff,
            self.verbose
        )
        
        io.save_fc(fc, path, self.output)
