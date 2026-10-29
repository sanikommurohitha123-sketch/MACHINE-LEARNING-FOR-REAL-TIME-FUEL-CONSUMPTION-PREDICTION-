from django.db import models

# Create your models here.
class PredictionLog(models.Model):
    year = models.IntegerField()
    make = models.IntegerField()
    model = models.IntegerField()
    vehicle_class = models.IntegerField()
    engine_size = models.FloatField()
    cylinders = models.IntegerField()
    transmission = models.IntegerField()
    fuel_type = models.IntegerField()
    hwy = models.FloatField()
    comb = models.FloatField()
    predicted_value = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)