from django.contrib import admin
from django.urls import path
from Prediction import views
from backend import diabetes


urlpatterns = [
    path("",views.base, name="Let's predict diabetes"),
    path("Home",views.Home, name='Home'),
    path("Blog",views.Blog, name='Blog'),
    path("Predict",views.Predict, name='Predict'),
    path("predict_diabetes",diabetes.predict_diabetes, name='predict_diabetes'),
    path("contact",views.contact, name='contact'),
    path("About",views.About, name='About us'),
    path("dietcharts",views.dietcharts, name='DietChart'),
    path("exr",views.Blog, name='exr'),
]
