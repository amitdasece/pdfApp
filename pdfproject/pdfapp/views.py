from django.shortcuts import render
<<<<<<< Updated upstream

# Create your views here.
=======
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import IndexDoc
from .serializers import IndexDocSerializer, RetrieveDocSerializer, CompanySerializer
from django.shortcuts import get_object_or_404
from .models import Company, FileUpload
 
class IndexDocView(APIView):
    def post(self, request):
        serializer = IndexDocSerializer(data=request.data)
        if serializer.is_valid():
            db_name = serializer.validated_data.get('db_name')
            company = serializer.validated_data.get('company_id')
            # Get the latest file uploaded by the company
            file_upload = company.file_uploads.latest('uploaded_at')
            indexer = IndexDoc()
            indexer.index(db_name=db_name, file_path=file_upload.file.path)
            return Response({"message": "Document indexed successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
 
class RetrieveDocView(APIView):
    def post(self, request):
        serializer = RetrieveDocSerializer(data=request.data)
        if serializer.is_valid():
            query = serializer.validated_data.get('query')
            db_type = serializer.validated_data.get('db_type')
            company = serializer.validated_data.get('company_id')
            # Get the latest file uploaded by the company
            print("NAME DB------>>>>>>companyId",db_type,company)
            file_upload = company.file_uploads.latest('uploaded_at')
            print("NAME DB-->>",db_type,file_upload.file.path)
            retriever = IndexDoc()
            result = retriever.retrieve(query, db_name="chroma_db_{}".format(company), file_path=file_upload.file.path)
            print("Result-->>", company,result)
            return Response({"answer": result}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
 
class CompanyListView(APIView):
    def get(self, request):
        companies = Company.objects.all()
        serializer = CompanySerializer(companies, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
 
class CompanyDetailView(APIView):
    def get(self, request, company_id):
        try:
            company = Company.objects.get(pk=company_id)
            serializer = CompanySerializer(company)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Company.DoesNotExist:
            return Response({'error': 'Company not found'}, status=status.HTTP_404_NOT_FOUND)
        
class SearchCompaniesView(APIView):
    def get(self, request, *args, **kwargs):
        query = request.GET.get('q', '')
        if query:
            companies = Company.objects.filter(name__icontains=query)
            serializer = CompanySerializer(companies, many=True)
            return Response(serializer.data)
        return Response([])  # Return an empty list if no query provided
>>>>>>> Stashed changes
