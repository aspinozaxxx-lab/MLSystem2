"""add columns preview_l and preview_s to images table

Revision ID: 2f05db38cede
Revises: 1aba2c61cc0c
Create Date: 2022-10-21 16:16:54.199813

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2f05db38cede'
down_revision = '1aba2c61cc0c'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(table_name='images', column_name='preview_url', new_column_name='preview_url_l')
    op.add_column('images', sa.Column('preview_url_s', sa.String(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    op.alter_column(table_name='images', column_name='preview_url_l', new_column_name='preview_url')
    op.drop_column('images', 'preview_url_s')
    # ### end Alembic commands ###
