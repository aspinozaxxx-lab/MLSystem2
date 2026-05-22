ARG CI_REGISTRY
FROM ${CI_REGISTRY}/nspd/geoalert/docker/ubuntu-gdal-libspatialindex:1.0.0
#FROM registry.gitlab.com/geoalert-projects/geoalert-docker/ubuntu-gdal-libspatialindex:0438e6fd
ARG PIP_SOURCE=https://pypi.org/simple/


ARG PIP_SOURCE=https://pypi.org/simple/
ARG CI_SERVER_HOST

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --index-url=$PIP_SOURCE --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY run.sh /code/run.sh
COPY wd.yaml /code/wd.yaml

ENV PYTHONPATH "${PYTHONPATH}:/code/app/"

EXPOSE 8080

CMD ["sh", "run.sh"]
