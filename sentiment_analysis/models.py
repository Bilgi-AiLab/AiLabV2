import json

from django.db import models

# Create your models here.
from project.models import Project

class Report(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='sentiment_project')
    algorithm = models.CharField(max_length=100)
    all_data = models.TextField()

    def get_output(self):
        return json.loads(self.all_data)


    def polarity_value(self):
        try:
            return self.get_output()["polarity_value"]
        except KeyError:
            return None
    
    def negative_doc_count(self):
        try:
            return self.get_output()["negative_doc_count"]
        except KeyError:
            return None
        
    def positive_doc_count(self):
        try:
            return self.get_output()["positive_doc_count"]
        except KeyError:
            return None
        
    def neutral_doc_count(self):
        try:
            return self.get_output()["neutral_doc_count"]
        except KeyError:
            return None
    
    def detailed_scores(self):
        try:
            return self.get_output()["detailed_scores"]
        except KeyError:
            return None
