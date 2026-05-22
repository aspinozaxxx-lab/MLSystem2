ARG CI_REGISTRY
FROM ${CI_REGISTRY}/nspd/geoalert/docker/dataloader-base:1.0.0

# copy external modules
COPY ./modules /modules

# install external modules
WORKDIR /modules/xyz_loader
RUN pip install --no-dependencies .

WORKDIR /modules/nspd_loader
RUN pip install --no-dependencies .

# copy app
COPY ./app /app

# set project workdir
WORKDIR /

# start applictaion
CMD ["python3", "-m", "app.worker"]
