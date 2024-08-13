from django.contrib import admin
 
# Register your models here.
from .models import FileUpload,Company
 
# Register your models here.
admin.site.register(FileUpload)
admin.site.register(Company)
# admin.site.register(Industry)