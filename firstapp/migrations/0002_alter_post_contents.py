# Generated by Django 4.1 on 2022-08-12 09:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('firstapp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='post',
            name='contents',
            field=models.TextField(null=True),
        ),
    ]