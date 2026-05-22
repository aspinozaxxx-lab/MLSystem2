# Urban
#### Build a pipeline for geospatial data processing.

#### 1. Requierments  

 - rasterio >= 1.0.0
 - aeronet == 0.0.17
 - shapely >= 1.6.4
 
#### 2. Installation  
From source
```bash
git clone git@gitlab.com:aeronetlab-sk-team/urban.git
cd urban
pip install .
```

#### 3. Quick start

```python

from keras.models import load_model

from urban.base import Compose
from urban.models import BinarySegmentationModel
from urban import Segmentation, LoadOSMBuildingsLike, Dedupe, FilterSmallObjects, SimplifyAsRectangles


# loading keras model
model = load_model('path/to/model.h5', compile=False)

# defining model for remote sensing data
rsd_model = BinarySegmentationModel(
    model=model,
    input_rasters=['RED', 'GRN', 'BLU'],
    output_labels=['100'], # building class
    res=(0.3, 0.3), 
    crs='utm',
    threshold=0.5,
)

# building a pipeline
bricks = [

    # make binary segmentation of buildings
    Segmentation(rsd_model, sample_size=(1024, 1024)),
    
    # make some post processing with generated vector markup
    FilterSmallObjects('100', 1000, output='100'),
    SimplifyAsRectangles('100', output='100'),
    
    # load OSM buildings for same extent
    LoadOSMBuildingsLike('100', output='100osm'),
    
    # merge predicted buildings with OSM
    Dedupe('100', prior='100osm', merge_threshold=0.6),
]

segmentation_pipeline = Compose(bricks)

# process data
data = '/path/to/data'
segmentation_pipeline(data)

```
#### 4. Tests  

To run tests put 'B08.tif' file in the folder tests/data/  
File link: https://filebrowser.aeronetlab.space/s/lhk1t2j3C9gIoFl
