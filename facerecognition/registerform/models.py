from django.db import models
import datetime
from django.contrib.auth.hashers import check_password as django_check_password
class User(models.Model):
    uid=models.CharField(primary_key=True,max_length=10)
    uname=models.CharField(max_length=100,default=None)
    uemail=models.EmailField(unique=True)
    ucontact=models.CharField(max_length=10,unique=True)
    upassword= models.CharField(max_length=256)
    access=models.CharField(max_length=1,default=0)
    class Meta:
        db_table='user'
class Markatt(models.Model):
    userid=models.CharField(max_length=10,default=None)
    name=models.CharField(max_length=100)
    email=models.EmailField(default=None)
    contact=models.CharField(max_length=10,default=None)
    time=models.TimeField(null=True, blank=True)
    date=models.DateField(default=datetime.date.today)
    status=models.CharField(max_length=8,default="inactive")
    class Meta:
        db_table="markatt"
'''
class Attendance(models.Model):
    userid=models.CharField(max_length=10,default=None)
    name=models.CharField(max_length=100)
    email=models.EmailField(unique=True)
    contact=models.CharField(max_length=10,unique=True)
    time=models.TimeField(null=True, blank=True)
    date=models.DateField(default=datetime.date.today)
    status=models.CharField(max_length=8,default="inactive")
    class Meta:
        db_table="attendance"
'''