utest:
	docker build --no-cache --target testing --build-arg GITLAB_ACCESS_TOKEN= -t cog-builder/test -f Dockerfile .
	docker run --rm cog-builder/test python3 -m pytest ./tests/unit
	docker rmi cog-builder/test
itest:
	docker build --target testing --build-arg GITLAB_ACCESS_TOKEN= -t cog-builder/test -f Dockerfile .
	docker compose up -d minio
	docker compose run --rm app python3 -m pytest -s -vv ./tests/it
	docker compose down
	docker rmi cog-builder/test
test:
	docker build --target testing --build-arg GITLAB_ACCESS_TOKEN= -t cog-builder/test -f Dockerfile .
	docker compose up -d minio
	docker compose run --rm app python3 -m pytest ./tests/it ./tests/unit
	docker compose down
	docker rmi cog-builder/test

