from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import IndexDoc
from .serializers import IndexDocSerializer, RetrieveDocSerializer
# from .helper import format_response
 
class IndexDocView(APIView):
    def post(self, request):
        print("hello",request)
        serializer = IndexDocSerializer(data=request.data)
        if serializer.is_valid():
            db_name = serializer.validated_data.get('db_name')
            indexer = IndexDoc()
            indexer.index(db_name=db_name)
            return Response({"message": "Document indexed successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
 
class RetrieveDocView(APIView):
    def post(self, request):
        serializer = RetrieveDocSerializer(data=request.data)
        if serializer.is_valid():
            query = serializer.validated_data.get('query')
            db_type = serializer.validated_data.get('db_type')
            retriever = IndexDoc()
            result = retriever.retrieve(query, db_type=db_type)
            print("-->>>result",result)
            # formatted_text = format_response(result)
            # print("-->>>text",formatted_text)
            return Response({"answer": result}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)