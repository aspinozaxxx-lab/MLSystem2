# 1. Build or download image
## 1.1 Download from gitlab
Login to gitlab registry
`docker login registry.hub.nspd.rosreestr.gov.ru` and when prompted, enter your gitlab login and password

Pull the image
`docker pull registry.hub.nspd.rosreestr.gov.ru/nspd/geoalert/mapflow/metrics:local`

## 1.2 Build locally
`docker build -t registry.hub.nspd.rosreestr.gov.ru/nspd/geoalert/mapflow/metrics:local -f Dockerfile.local .`


# 2. Prepare data
Folder with files will be mounted into the container. Structure:

`root/NSPD_TEST_DATA/<task>/<sample>/[image.tif, gt.geojson, pred.geojson]`

# 3. Launch the container
`docker run --rm -p 8888:8888 -it -v <root data folder>:/home/user/data --name metrics -d registry.hub.nspd.rosreestr.gov.ru/nspd/geoalert/mapflow/metrics:local`

# 4. Connect to jupyter lab
Open `localhost:8888/lab` in browser. Password is `aeronet15`

# 5. Run notebooks
`/home/user/Metrics_NSPD_pixelwise.ipynb`
`/home/user/Metrics_NSPD_objectwise.ipynb`

Change folder path and task name if necessary. 