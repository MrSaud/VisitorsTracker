"""Rename CountEvent to Crossing.

Hand-written so the table is renamed rather than dropped and recreated:
an automatic run would see a deleted model and a new one, and any recorded
history would be lost. "Crossing" also frees the name Event for the thing
an administrator schedules.
"""

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("counter", "0001_initial"),
    ]

    operations = [
        migrations.RenameModel(
            old_name="CountEvent",
            new_name="Crossing",
        ),
    ]
