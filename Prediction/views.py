from django.shortcuts import render, HttpResponse

# Create your views here.

def base(request):
    context = {
        "variable":"start test"
    }
    return render(request,'base.html', context)
    #return HttpResponse("This is Prediction")  #when we use string then we use http response 
def Home(request):
    return render(request,'Home.html')
def Blog(request):
     return render(request,'Blog.html')
def Predict(request):
     return render(request,'Predict.html')
def contact (request):
     return render(request,'contact.html')
def About (request):
     return render(request,'About.html')
def dietcharts (request):
     return render(request,'dietcharts.html')




