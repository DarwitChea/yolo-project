"""Fresh migration with attendance schema update

Revision ID: 2107d1afd591
Revises: 
Create Date: 2025-06-09 20:31:20.357096

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2107d1afd591'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('attendance', recreate='always') as batch_op:
        # Recreate the table without the 'present' column
        batch_op.drop_column('present')
        # Add the unique constraint
        batch_op.create_unique_constraint('unique_student_session', ['student_id', 'session'])



def downgrade():
    with op.batch_alter_table('attendance', recreate='always') as batch_op:
        batch_op.drop_constraint('unique_student_session', type_='unique')
        batch_op.add_column(sa.Column('present', sa.BOOLEAN(), nullable=True))

