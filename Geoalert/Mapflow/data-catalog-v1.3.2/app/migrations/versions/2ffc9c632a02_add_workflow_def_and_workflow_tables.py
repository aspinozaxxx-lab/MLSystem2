"""add workflow_def and workflow tables

Revision ID: 2ffc9c632a02
Revises: 5c4db9ad27d2
Create Date: 2022-12-22 15:48:40.217974

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '2ffc9c632a02'
down_revision = '5c4db9ad27d2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('workflow',
                    sa.Column('image_id', postgresql.UUID(as_uuid=True), nullable=False),
                    sa.Column('we_id', sa.Integer(), nullable=True),
                    sa.Column('status', sa.Enum('UNPROCESSED',
                                                'IN_PROGRESS',
                                                'OK',
                                                'FAILED',
                                                name='workflow_status_enums'),
                              nullable=False),
                    sa.PrimaryKeyConstraint('image_id')
                    )

    op.create_table('workflow_def',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('yaml', sa.TEXT(), nullable=True),
                    sa.PrimaryKeyConstraint('id')
                    )
    op.add_column('images', sa.Column('cog_link', sa.String(), nullable=True))
    op.add_column('mosaics', sa.Column('cog_link', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_table('workflow_def')
    op.drop_table('workflow')
    op.drop_column('images', 'cog_link')
    op.drop_column('mosaics', 'cog_link')
    sa.Enum(name='workflow_status_enums').drop(op.get_bind(), checkfirst=False)
