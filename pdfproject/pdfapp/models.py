import os
from django.utils.deconstruct import deconstructible
from django.db import models
from django.core.exceptions import ValidationError
from .utils import IndexDoc
 
 
@deconstructible
class PathAndRename:
    def __init__(self, path):
        self.sub_path = path
 
    def __call__(self, instance, filename):
        ext = filename.split('.')[-1]
        # set filename as the company name
        filename = '{}.{}'.format(instance.company.name, ext)
        # return the whole path to the file
        return os.path.join(self.sub_path, filename)
 
 
def validate_file_size(value):
    filesize = value.size
    if filesize > 104857600:  # 100MB in bytes
        raise ValidationError("The maximum file size that can be uploaded is 100MB")
    return value
 
class Company(models.Model):
    name = models.CharField(max_length=255, unique=True)
    website = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name
 
class FileUpload(models.Model):
    file = models.FileField(upload_to=PathAndRename('uploads/'), validators=[validate_file_size])
    uploaded_at = models.DateTimeField(auto_now_add=True)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='file_uploads')
 
    def __str__(self):
        return self.file.name
 
    def save(self, *args, **kwargs):
        self.file.name = '{}.{}'.format(self.company.name, self.file.name.split('.')[-1])
        super().save(*args, **kwargs)
        indexer = IndexDoc()
        db_name = "chroma_db_" + self.company.name
        print("Company Name:", self.company.name)
        print("Database Name:", db_name)
        print("File Path:", self.file.path)
        indexer.index(db_name=db_name, file_path=self.file.path)