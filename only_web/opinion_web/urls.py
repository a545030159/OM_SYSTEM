"""opinion_web URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from . import views

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('index/', views.index2),
    # path('index_look/', views.index),
    path('post/', views.post),
    path('post_1/', views.post_1),
    path('', views.index1),
    path('mine/', views.mine),
    path('mine_1/', views.mine_1),
    path('upload/', views.upload),
    path('upload_1/', views.upload_1),
    path('load/', views.load),
    path('crawlers/', views.crawlers)
]
