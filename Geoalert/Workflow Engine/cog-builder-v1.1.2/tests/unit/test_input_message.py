from cog_builder.app.message import CogBuilderInputMessage, CogBuilderInputData, CogBuilderOutputData
from shapely.geometry import Polygon

def test_input_message():
    data = CogBuilderInputData(raster_source="s3://bucket/path/to/file.tif",
                               compress="WEBP",
                               channels="1,2,3",
                               aoi={"type": "Polygon", "coordinates": [[[9900, 20100],
                                                                     [10000, 20200],
                                                                     [10100, 20100],
                                                                     [9900, 20100]]]})
    message = CogBuilderInputMessage(task_id = 42,
                                     runcheck_url = 'https://runcheck',
                                     input=data,
                                     output=CogBuilderOutputData(target_uri="s3://bucket/path/to/output.tif"))

    assert message.input.raster_source == "s3://bucket/path/to/file.tif"
    assert message.input.compress == "WEBP"
    assert message.input.channels == [1, 2, 3]
    assert message.input.aoi.is_valid
    assert isinstance(message.input.aoi, Polygon)