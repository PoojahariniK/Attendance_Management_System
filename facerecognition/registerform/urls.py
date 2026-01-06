from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
'''
urlpatterns=[
    path('',views.registerform,name="registerform"),
]'''
urlpatterns = [
    path('',views.home,name="home"),
    path('attendance/',views.attendance,name='attendance'),
    path('userreg/', views.userreg, name='userreg'),
    path('userreg/insertuser/', views.insertuser, name='insertuser'),
    path('userlogin/', views.userlogin, name='userlogin'),
    path('userlogin/getback/', views.getback, name='getback'),
    path("userlist/",views.userlist,name='userlist'),
    path("userrecord/<str:id>/",views.userrecord,name="userrecord"),
    #path("noatt/",views.userlogin,name="noatt"),
    path("back/",views.back,name="back"),#back to employee list
    path("editname/<str:name>/",views.editname,name="editname"),
    path("editvalue/<str:name>",views.editvalue,name="editvalue"),
    path("editname_em/<str:name>/",views.editname_em,name="editname_em"),
    path("editvalue_em/<str:name>",views.editvalue_em,name="editvalue_em"),
    path("editcontact/<str:contact>",views.editcontact,name="editcontact"),
    path("editvalue_contact/<str:contact>",views.editvalue_contact,name="editvalue_contact"),
    path("editcontact_em/<str:contact>",views.editcontact_em,name="editcontact_em"),
    path("editvalue_contact_em/<str:contact>",views.editvalue_contact_em,name="editvalue_contact_em"),
    #path('userin/<str:username>/', views.userin, name='userin'),
    #path('accounts/login/', auth_views.LoginView.as_view(template_name='registerform/userin.html'), name='login'),
]
urlpatterns+=staticfiles_urlpatterns()