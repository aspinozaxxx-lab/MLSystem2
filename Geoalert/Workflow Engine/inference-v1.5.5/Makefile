utest:
	docker build --no-cache --target testing --build-arg GITLAB_ACCESS_TOKEN= -t inference/test -f Dockerfile .
	docker run --rm inference/test python3 -m pytest -vv ./tests/unit
	docker rmi inference/test
itest:
	docker build --target testing --build-arg GITLAB_ACCESS_TOKEN= -t inference/test -f Dockerfile .
	docker compose up -d minio
	docker compose run --rm app python3 -m pytest -s -vv ./tests/it
	docker compose down
	docker rmi inference/test
test:
	docker build --target testing --build-arg GITLAB_ACCESS_TOKEN= -t inference/test -f Dockerfile .
	docker compose up -d minio
	docker compose run --rm app python3 -m pytest ./tests/it ./tests/unit
	docker compose down
	docker rmi inference/test

