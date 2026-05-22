### Tool for downloading raster tiles

#### Installation

1. Using Pip
```bash
$ pip install git+https://gitlab.com/aeronetlab-sk-team/dataloader-project/dataloader.git
```

2. Using Docker
```bash
$ git clone https://gitlab.com/aeronetlab-sk-team/dataloader-project/dataloader
$ cd dataloader
$ docker build -t dataloader:dev .
```

#### Usage

1. Start container
```bash
$ docker run -it -rm -v ~/data:/data dataloader:dev bash
```

##### Command-line interface

2. Execute command in container
```bash
$ maploader download [OPTIONS] URL
```

3. To know more about options
```bash
$ maploader download --help

```

##### Python code example

```python
from maploader import download

URL = ''
ZOOM = 18

output_fp = '/path/to/output.tif'
geometry = {
    'type': 'Polygon',
    'coordinates': ...,
}

download(URL, ZOOM, geometry, output_fp)
```

### Run tests

Only unit tests:
```bash
$ chmod +x ./test.sh
$ ./test.sh
```

Only integration tests:
```bash
$ chmod +x ./test.sh
$ ./test.sh it
```

All tests:
```bash
$ chmod +x ./test.sh all
$ ./test.sh it
```