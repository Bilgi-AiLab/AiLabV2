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


    def summac_score(self):
        try:
            return self.get_output()["summac_score"]
        except KeyError:
            return None
    
    def summary(self):
        try:
            return self.get_output()["summary"]
        except KeyError:
            return None

