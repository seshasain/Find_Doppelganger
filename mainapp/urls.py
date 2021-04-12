from django.urls import path
from . import views
urlpatterns=[
    path('',views.homepage),
    path('homepage',views.homepage),
    path('results', views.results),
    path('upload', views.upload),
]