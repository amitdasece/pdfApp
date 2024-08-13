from rest_framework import serializers
from .models import Company, FileUpload
 
class IndexDocSerializer(serializers.Serializer):
    db_name = serializers.CharField()
    company_id = serializers.PrimaryKeyRelatedField(queryset=Company.objects.all())
 
class RetrieveDocSerializer(serializers.Serializer):
    query = serializers.CharField()
    db_type = serializers.CharField()
    company_id = serializers.PrimaryKeyRelatedField(queryset=Company.objects.all())
 
class CompanySerializer(serializers.ModelSerializer):
    class Meta:
        model = Company
        fields = ['id', 'name']