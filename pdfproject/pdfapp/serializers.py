from rest_framework import serializers
 
class IndexDocSerializer(serializers.Serializer):
    db_name = serializers.CharField(max_length=100, default="chromadb")
 
class RetrieveDocSerializer(serializers.Serializer):
    query = serializers.CharField()
    db_type = serializers.CharField(max_length=100, default="chroma_db")