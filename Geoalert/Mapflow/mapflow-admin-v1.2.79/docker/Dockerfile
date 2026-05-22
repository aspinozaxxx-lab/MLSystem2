ARG SOURCE_IMAGE=nginx:1.23-alpine
FROM $SOURCE_IMAGE

ARG REACT_APP_VERSION
ENV REACT_APP_VERSION=$REACT_APP_VERSION

ARG TIER=""

COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY build-$TIER /var/www/mapflow/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
