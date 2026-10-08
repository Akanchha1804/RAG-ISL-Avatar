"""clip inventory tables (sentences / signs / phrases)

Revision ID: 002
Revises: 001
Create Date: 2026-10-07
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'sentence_clips',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('text_norm', sa.Text, nullable=False, unique=True),
        sa.Column('text_display', sa.Text),
        sa.Column('glosses', sa.Text),
        sa.Column('canonical_gloss', sa.Text, nullable=True),
        sa.Column('landmark_file', sa.String(500)),
        sa.Column('status', sa.String(20), server_default='ready'),
        sa.Column('video_count', sa.Integer),
        sa.Column('detection_rate', sa.Float),
    )
    op.create_index('idx_sentence_clips_status', 'sentence_clips', ['status'])

    op.create_table(
        'sign_clips',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('gloss', sa.String(100), nullable=False, unique=True),
        sa.Column('uid', sa.String(100), nullable=True),
        sa.Column('category', sa.String(100), nullable=True),
        sa.Column('duration', sa.Float, nullable=True),
        sa.Column('landmark_clip', sa.String(200), nullable=True),
        sa.Column('landmark_file', sa.String(500), nullable=True),
        sa.Column('source', sa.String(30), server_default='cislr'),
    )
    op.create_index('idx_sign_clips_source', 'sign_clips', ['source'])

    op.create_table(
        'phrase_clips',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('phrase', sa.Text, nullable=False, unique=True),
        sa.Column('token_count', sa.Integer),
        sa.Column('source', sa.String(30), server_default='composed'),
        sa.Column('note', sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_table('phrase_clips')
    op.drop_index('idx_sign_clips_source')
    op.drop_table('sign_clips')
    op.drop_index('idx_sentence_clips_status')
    op.drop_table('sentence_clips')
