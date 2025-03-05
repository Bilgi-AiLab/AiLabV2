from django.urls import path

from . import views

urlpatterns = [
    path('api/chatbot', views.chatbot, name='chatbot_response'),
]