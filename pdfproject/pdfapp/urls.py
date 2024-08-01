from django.urls import path
from .views import IndexDocView, RetrieveDocView
 
urlpatterns = [
    path('index/', IndexDocView.as_view(), name='index-doc'),
    path('retrieve/', RetrieveDocView.as_view(), name='retrieve-doc'),
]