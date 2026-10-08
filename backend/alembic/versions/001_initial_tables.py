"""initial tables

Revision ID: 001
Revises:
Create Date: 2026-09-23

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'translation_history',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('input_text', sa.Text, nullable=False),
        sa.Column('output_glosses', sa.Text),
        sa.Column('matched_sentence', sa.Text),
        sa.Column('similarity', sa.Float),
        sa.Column('landmark_file', sa.String(500)),
        sa.Column('method', sa.String(50)),
        sa.Column('stt_transcript', sa.Text),
        sa.Column('stt_language', sa.String(20)),
        sa.Column('stt_duration', sa.Float),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_translation_history_created_at', 'translation_history', ['created_at'])
    op.create_index('idx_translation_history_method', 'translation_history', ['method'])


def downgrade() -> None:
    op.drop_index('idx_translation_history_method')
    op.drop_index('idx_translation_history_created_at')
    op.drop_table('translation_history')
