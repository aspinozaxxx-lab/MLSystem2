FROM registry.gitlab.com/geoalert-projects/geoalert-docker/ubuntu-gdal-libspatialindex:0438e6fd

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

ENV PYTHONPATH "${PYTHONPATH}:/code/app/"

EXPOSE 8000