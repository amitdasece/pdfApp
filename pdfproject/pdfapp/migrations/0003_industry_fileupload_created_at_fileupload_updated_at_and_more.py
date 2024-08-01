# Generated by Django 5.0.7 on 2024-07-30 07:25

import django.core.validators
import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pdfapp', '0002_company_fileupload_company'),
    ]

    operations = [
        migrations.CreateModel(
            name='Industry',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, unique=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.AddField(
            model_name='fileupload',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='fileupload',
            name='updated_at',
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AddField(
            model_name='fileupload',
            name='year',
            field=models.IntegerField(default=2024, validators=[django.core.validators.MinValueValidator(1900), django.core.validators.MaxValueValidator(2024)]),
        ),
        migrations.AddField(
            model_name='fileupload',
            name='industry',
            field=models.ForeignKey(default='', on_delete=django.db.models.deletion.CASCADE, related_name='file_uploads_industry', to='pdfapp.industry'),
            preserve_default=False,
        ),
    ]
