from django.contrib import admin
from django.urls import path, include
import chatbot.views as views

urlpatterns = [
    path('query', views.getData),
]
