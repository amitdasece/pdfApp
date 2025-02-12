# Generated by Django 5.0.7 on 2024-07-29 11:04

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pdfapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Company',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, unique=True)),
                ('website', models.URLField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.AddField(
            model_name='fileupload',
            name='company',
            field=models.ForeignKey(default='', on_delete=django.db.models.deletion.CASCADE, related_name='file_uploads', to='pdfapp.company'),
            preserve_default=False,
        ),
    ]
