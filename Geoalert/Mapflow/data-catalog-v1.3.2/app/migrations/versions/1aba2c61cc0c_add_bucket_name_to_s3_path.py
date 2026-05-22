"""add-bucket-name-to-s3-path

Revision ID: 1aba2c61cc0c
Revises: 42ad8967260b
Create Date: 2022-09-07 16:17:47.372349

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '1aba2c61cc0c'
down_revision = '42ad8967260b'
branch_labels = None
depends_on = None

BUCKET_PREFIX = "s3://users-data/"


def upgrade() -> None:
    # for all entries that contain s3 path (image.image_url, image.preview_url, mosaic_mosaic_url)
    # add s3://{MINIO_BUCKET}/ to the values
    connection = op.get_bind()

    # Migrate data
    t_data = sa.Table(
        'images',
        sa.MetaData(),
        sa.Column('id', sa.String(32)),
        sa.Column('image_url', sa.Unicode(length=255)),
        sa.Column('preview_url', sa.Unicode(length=255))
        )
    results = connection.execute(sa.select([
        t_data.c.id,
        t_data.c.image_url,
        t_data.c.preview_url
        ])).fetchall()
    # Iterate over all selected data tuples.
    for id_, image_url, preview_url in results:
        if not image_url.startswith("s3://"):
            image_url = BUCKET_PREFIX + image_url
        if not preview_url.startswith("s3://"):
            preview_url = BUCKET_PREFIX + preview_url
        # Update the new columns.
        connection.execute(t_data.update().where(t_data.c.id == id_).values(image_url=image_url,
                                                                            preview_url=preview_url))

    # Migrate mosaic
    t_data = sa.Table(
        'mosaics',
        sa.MetaData(),
        sa.Column('id', sa.String(32)),
        sa.Column('mosaic_url', sa.Unicode(length=255))
        )
    results = connection.execute(sa.select([
        t_data.c.id,
        t_data.c.mosaic_url,
        ])).fetchall()
    # Iterate over all selected data tuples.
    for id_, mosaic_url in results:
        if not mosaic_url.startswith("s3://"):
            mosaic_url = BUCKET_PREFIX + mosaic_url
        connection.execute(t_data.update().where(t_data.c.id == id_).values(mosaic_url=mosaic_url))


def downgrade() -> None:
    connection = op.get_bind()

    # Migrate data
    t_data = sa.Table(
        'images',
        sa.MetaData(),
        sa.Column('id', sa.String(32)),
        sa.Column('image_url', sa.Unicode(length=255)),
        sa.Column('preview_url', sa.Unicode(length=255))
    )
    results = connection.execute(sa.select([
        t_data.c.id,
        t_data.c.image_url,
        t_data.c.preview_url
    ])).fetchall()
    # Iterate over all selected data tuples.
    for id_, image_url, preview_url in results:
        if image_url.startswith(BUCKET_PREFIX):
            image_url = image_url[len(BUCKET_PREFIX):]
        if preview_url.startswith(BUCKET_PREFIX):
            preview_url = preview_url[len(BUCKET_PREFIX):]
        # Update the new columns.
        connection.execute(t_data.update().where(t_data.c.id == id_).values(image_url=image_url,
                                                                            preview_url=preview_url))

    # Migrate mosaic
    t_data = sa.Table(
        'mosaics',
        sa.MetaData(),
        sa.Column('id', sa.String(32)),
        sa.Column('mosaic_url', sa.Unicode(length=255))
    )
    results = connection.execute(sa.select([
        t_data.c.id,
        t_data.c.mosaic_url,
    ])).fetchall()
    # Iterate over all selected data tuples.
    for id_, mosaic_url in results:
        if mosaic_url.startswith(BUCKET_PREFIX):
            mosaic_url = mosaic_url[len(BUCKET_PREFIX):]
        connection.execute(t_data.update().where(t_data.c.id == id_).values(mosaic_url=mosaic_url))
