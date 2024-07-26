from django.db import models
from django.core.exceptions import ValidationError
 
def validate_file_size(value):
    filesize = value.size
    if filesize > 104857600:  # 100MB in bytes
        raise ValidationError("The maximum file size that can be uploaded is 100MB")
    return value
 
 
class FileUpload(models.Model):
    file = models.FileField(upload_to='uploads/', validators=[validate_file_size])
    uploaded_at = models.DateTimeField(auto_now_add=True)
 
    def __str__(self):
        return self.file.name