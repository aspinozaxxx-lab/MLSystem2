run:
	docker compose up -d app auth_server minio db
	# sleep 5
	# docker compose exec -w /code/app app alembic upgrade heads

build:
	docker compose build app

test:
	MAX_IMAGE_SIZE_PIXELS=2000 docker compose up -d app auth_server minio db
	sleep 11
	# docker compose exec -w /code/app app alembic upgrade heads
	sleep 2
	docker build -t data-catalog:test -f Dockerfile.test .
	docker run -it --network="host" --rm data-catalog:test pytest tests/it
	docker compose down
	docker image rm data-catalog:test data-catalog-app

unittest:
	docker build -t data-catalog:test -f Dockerfile.test .
	docker run -it --rm data-catalog:test pytest tests/unit
	docker image rm data-catalog:test

stop:
	docker compose down


restart:
	docker compose down
	docker image rm data-catalog-app
	docker compose up -d --build app auth_server minio db
	# sleep 5
	# docker compose exec -w /code/app app alembic upgrade heads
