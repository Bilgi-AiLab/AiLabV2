from django.urls import path

from . import views

urlpatterns = [
    path('<int:pk>/sentiment-analysis', views.sentiment_algorithms, name='sentiment_algorithms'),
    path('<int:pk>/sentiment-analysis/<str:algorithm>', views.apply_sentiment_algorithm, name='apply_sentiment_algorithm'),
    path('<int:project_pk>/sentiment-analysis/<str:algorithm>/<int:report_pk>', views.view_sentiment_report, name='view_sentiment_report'),
    path('<int:project_pk>/sentiment-analysis/<str:algorithm>/<int:report_pk>/remove', views.remove_sentiment_report,
         name='remove_sentiment_report'),
]