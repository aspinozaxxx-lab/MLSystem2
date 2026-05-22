"""

Revision ID: c3ddf139858c
Revises: 2f05db38cede
Create Date: 2022-10-20 13:35:25.810455

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c3ddf139858c'
down_revision = '2f05db38cede'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(table_name='users', column_name='memory_used', type_=sa.types.BigInteger)
    op.alter_column(table_name='users', column_name='memory_limit', type_=sa.types.BigInteger)


def downgrade() -> None:
    op.alter_column(table_name='users', column_name='memory_used', type_=sa.types.Integer)
    op.alter_column(table_name='users', column_name='memory_limit', type_=sa.types.Integer)
