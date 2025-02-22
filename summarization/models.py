import json

from django.db import models

# Create your models here.
from project.models import Project

class Report(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='summarization_project')
    algorithm = models.CharField(max_length=100)
    all_data = models.TextField()

    def get_output(self):
        return json.loads(self.all_data)


    def rouge1(self):
        try:
            return self.get_output()["rouge1"]
        except KeyError:
            return None
    
    def rouge2(self):
        try:
            return self.get_output()["rouge2"]
        except KeyError:
            return None
        
    def rougeL(self):
        try:
            return self.get_output()["rougeL"]
        except KeyError:
            return None
    
    def summary(self):
        try:
            return self.get_output()["summary"]
        except KeyError:
            return None

