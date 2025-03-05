import json

from django.db import models

class Report(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    all_data = models.TextField()

    def get_output(self):
        return json.loads(self.all_data)
    
    def get_response(self):
        try:
            return self.get_output()["response"]
        except KeyError:
            return None

