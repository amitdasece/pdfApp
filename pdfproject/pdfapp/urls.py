from django.urls import path
from .views import IndexDocView, RetrieveDocView,CompanyListView,CompanyDetailView,SearchCompaniesView
 
urlpatterns = [
    path('index/', IndexDocView.as_view(), name='index-doc'),
    path('retrieve/', RetrieveDocView.as_view(), name='retrieve-doc'),
    path('companies/', CompanyListView.as_view(), name='company-list'),
    path('company/<int:company_id>/', CompanyDetailView.as_view(), name='company-detail'),
    path('search_companies/', SearchCompaniesView.as_view(), name='search_companies'),
]