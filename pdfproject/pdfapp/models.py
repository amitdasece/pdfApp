from django.db import models
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator, MaxValueValidator
from datetime import datetime
 
def validate_file_size(value):
    filesize = value.size
    if filesize > 104857600:  # 100MB in bytes
        raise ValidationError("The maximum file size that can be uploaded is 100MB")
    return value


    
class Industry(models.Model):
    name = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name
    
class Company(models.Model):
    name = models.CharField(max_length=255, unique=True)
    website = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name 
 
class FileUpload(models.Model):
    file = models.FileField(upload_to='uploads/', validators=[validate_file_size])
    uploaded_at = models.DateTimeField(auto_now_add=True)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='file_uploads')
    industry = models.ForeignKey(Industry, on_delete=models.CASCADE, related_name='file_uploads_industry')
    year = models.IntegerField(validators=[MinValueValidator(1900), MaxValueValidator(datetime.now().year)], default=datetime.now().year)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.file.name

