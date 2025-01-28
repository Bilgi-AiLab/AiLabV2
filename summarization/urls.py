from django.urls import path

from . import views

urlpatterns = [
    path('<int:pk>/summarization', views.summarization_algorithms, name='summarization_algorithms'),
    path('<int:pk>/summarization/<str:algorithm>', views.apply_summarization_algorithm, name='apply_summarization_algorithm'),
    path('<int:project_pk>/summarization/<str:algorithm>/<int:report_pk>', views.view_summarization_report, name='view_summarization_report'),
    path('<int:project_pk>/summarization/<str:algorithm>/<int:report_pk>/remove', views.remove_summarization_report,
         name='remove_summarization_report'),
]