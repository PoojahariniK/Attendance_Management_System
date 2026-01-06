from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from .models import User
from django.contrib.auth.hashers import make_password
import re
import os
from django.conf import settings
from django.core.files.base import ContentFile
#hanndling image
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from django.contrib.auth.hashers import check_password #CHECKING PASSWORD
import logging
from tensorflow.keras.models import load_model# for loading model
import numpy as np
from .models import User,Markatt
import datetime
'''=====================================================================HOME=========================================================================='''
def home(request):
    return render(request,"registerform/home.html",{})
'''===================================================================REGISTER -PAGE==================================================================='''
def userreg(request):
    return render(request, "registerform/userreg.html", {})
'''==================================================================REGISTERATION - IMAGE CAPTURE================================================================================='''
def capture_images_and_save(user_id):
    captured_images = []
    capture_count = 50  # Capture 50 images
    captured_count = 0

    def capture_image():
        nonlocal captured_count
        if captured_count < capture_count:
            ret, frame = cap.read()
            if ret:
                ret, buf = cv2.imencode(".jpg", frame)
                if ret:
                    # Create ContentFile with image data
                    captured_image = ContentFile(buf.tobytes(), f'user_{user_id}_{captured_count}.jpg')
                    captured_images.append(captured_image)
                    captured_count += 1
            root.after(500, capture_image)  # Capture image every 0.5 second
        else:
            root.destroy()

    def update_frame():
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            root.after(10, update_frame)
        else:
            messagebox.showerror("Error", "Failed to capture image")

    root = tk.Tk()
    root.title("Capture Image")
    root.attributes("-topmost", True)
    lmain = tk.Label(root)
    lmain.pack()
    label = tk.Label(root, text="Capturing images... Please stay in front of the camera.")
    label.pack(pady=5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access camera")
        root.destroy()
        return []

    update_frame()
    root.lift()
    root.after(1000, capture_image)  # Start capturing images after 1 second
    root.mainloop()

    cap.release()

    # Save captured images to user's directory in media folder
    user_directory = os.path.join(settings.MEDIA_ROOT, f'user_{user_id}')
    os.makedirs(user_directory, exist_ok=True)  # Create user directory if it doesn't exist

    for idx, img in enumerate(captured_images):
        img_name = f'user_{user_id}_{idx}.jpg'
        img_path = os.path.join(user_directory, img_name)
        with open(img_path, 'wb') as f:
            f.write(img.read())

    return captured_images
'''=======================================================================REGISTER -FUNCTION============================================================================'''
def insertuser(request):
    if request.method == 'POST':
        vuid = request.POST.get('tuid')
        vuname = request.POST.get('tuname')
        vuemail = request.POST.get('tuemail')
        vucontact = request.POST.get('tucontact')
        vupassword = request.POST.get('tupassword')
        vuconfirmpassword = request.POST.get('tuconfirmpassword')
        #Capture image using opencv
        errors = []
        # UID validation
        if not vuid.isdigit() or len(vuid) != 10:
            errors.append('Enter valid UID (10 digits).')

        # Email validation
        if not re.match(r'^[\w.-]+@[a-zA-Z\d.-]+\.[a-zA-Z]{2,}$', vuemail):
            errors.append('Enter a valid email address.')

        # Contact validation
        if not vucontact.isdigit() or len(vucontact) != 10:
            errors.append('Enter valid mobile number (10 digits).')

        # Password validation
        if not re.match(r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,16}$', vupassword):
            errors.append('Password must contain at least one uppercase letter, one lowercase letter, one digit, one special character, and be at least 8 characters long.')

        # Password confirmation
        if vupassword != vuconfirmpassword:
            errors.append('Passwords do not match.')

        # Check if user already exists
        if User.objects.filter(uid=vuid).exists():
            errors.append('User with this UID already exists.')
        if User.objects.filter(uemail=vuemail).exists():
            errors.append('User with this email already exists.')
        if User.objects.filter(ucontact=vucontact).exists():
            errors.append('User with this contact number already exists.')
        if errors:
            for error in errors:
                messages.error(request, error)
        
        captured_images = capture_images_and_save(vuid)
        errorimg=[]
        if not captured_images:
            errorimg.append('Failed to capture images.')
        else:
                # Create and save the new user if images are captured successfully
            hashed_password = make_password(vupassword)
            try:
                user = User(uid=vuid, uname=vuname, uemail=vuemail, ucontact=vucontact, upassword=hashed_password)
                user.save()
                messages.success(request, 'Registered Successfully.')
                return redirect('userreg')
            except Exception as e:
                    errorimg.append(str(e))
        # If there are errors, display messages and return to form
        if errorimg:
            for error in errorimg:
                messages.error(request, error)
            return render(request, 'registerform/userreg.html', {})
        

    # If not a POST request, just render the form again
    return render(request, 'registerform/userreg.html', {})

'''================================================================LOGIN PAGE================================================================================='''
def userlogin(request):
    return render(request, "registerform/userlogin.html", {})

'''=================================================================USER DISPLAY PAGE-EMPLOYEE ACCESS=============================================================================='''
def userin(request):#,username
    return render(request,"registerform/userin.html",{})#"user":username.uname
'''===========================================================USER RECORD - MANAGER ACCESS================================================================================='''
def userrecord(request,id):
    if Markatt.objects.filter(userid=id).exists():
        user_info=Markatt.objects.filter(userid=id)
        context={"users":user_info}
        return render(request,"registerform/userrecord.html",context)
    else:
        users=User.objects.get(uid=id)
        context={"users":users}
        return render(request,"registerform/noatt_manager.html",context)

'''=============================================================BACK TO EMPLOYEE LIST - MANAGER ACCESS============================================================================='''
def back(request):
    name=User.objects.all()
    context_list={"names":name}
    return render(request,"registerform/userlist.html",context_list)
'''=========================================================USER LIST-MANAGER ACCESS================================================================================='''
def userlist(request):
    return render(request,"registerform/userlist.html",{})
'''======================================================================LOGIN PAGE - FUNCTION==============================================================================='''
logger = logging.getLogger(__name__)
def getback(request):
    if request.method == 'POST':
        email = request.POST.get('lemail')
        password = request.POST.get('lpassword')
        logger.debug(f"Attempting login with email: {email}")

        #user = authenticate(request, uemail=email, upassword=password)
        if User.objects.filter(uemail=email).exists():
            passw=User.objects.get(uemail=email)
            accessval=passw.access
            if check_password(password,passw.upassword):
                if accessval=="1":
                    #messagebox.showinfo("access:",{accessval})
                    name=User.objects.filter()
                    context_list={"names":name}
                    return render(request,"registerform/userlist.html",context_list)
                elif Markatt.objects.filter(email=email).exists():
                    display_user=Markatt.objects.filter(email=email)
                    #check=User.objects.get(uemail=email).access
                    context={"users":display_user}#,"access":check
                    return render(request,"registerform/userin.html",context)
                else:
                    context={"users":passw}
                    return render(request,"registerform/noatt.html",context)
            else:
                messages.error(request,"invalid")
        else:
            messages.error(request, "Invalid email or password")
            logger.warning(f"Failed login attempt with email: {email}")
    return render(request, 'registerform/userlogin.html')
'''========================================================EDIT NAME- manager PAGE==============================================================================================='''
def editname(request,name):
    user_details=User.objects.get(uname=name)
    #print(user_details)
    context={"users":user_details}
    return render(request,"registerform/editname.html",context)
'''==============================================================EDIT NAME-MANAGER======================================================================================'''
def editvalue(request,name):
    if request.method=="POST":
        new_name=request.POST.get("editedname")
        #print(new_name)
        if new_name==name:
            messages.error(request,"Same as old name")
        elif User.objects.filter(uname=new_name).exists():
            messages.error(request,"Username already exists")
        else:
            users=User.objects.get(uname=name)
            #print(users)
            users.uname=new_name
            users.save()
            markatts=Markatt.objects.filter(name=name)
            for markatt in markatts:
                markatt.name=new_name
                markatt.save()
            context={"users":markatts}
            messages.success(request,"Edited successfully")
            return render(request,"registerform/userrecord.html",context)
    markatts=Markatt.objects.filter(name=name)
    context={"users":markatts}
    return render(request,"registerform/userrecord.html",context)
'''=============================================================EDIT CONTACT -MANAGER PAGE==========================================================================================='''
def editcontact(request,contact):
    user_details=User.objects.get(ucontact=contact)
    #print(user_details)
    context={"users":user_details}
    return render(request,"registerform/editcontact.html",context)
'''=============================================================EDIT CONTACT - MANAGER================================================================================'''
def editvalue_contact(request,contact):
    if request.method=="POST":
        new_contact=request.POST.get("editedcontact")
        #print(new_name)
        if new_contact==contact:
            messages.error(request,"Same as old contact")
        elif User.objects.filter(ucontact=new_contact).exists():
            messages.error(request,"Contact already exists")
        else:
            users=User.objects.get(ucontact=contact)
            #print(users)
            users.ucontact=new_contact
            users.save()
            markatts=Markatt.objects.filter(contact=contact)
            for markatt in markatts:
                markatt.contact=new_contact
                markatt.save()
            context={"users":markatts}
            messages.success(request,"Edited successfully")
            return render(request,"registerform/userrecord.html",context)
    markatts=Markatt.objects.filter(contact=contact)
    context={"users":markatts}
    return render(request,"registerform/userrecord.html",context)

'''=============================================================EDIT CONTACT-EMPLOYEE PAGE==========================================================================================='''
def editcontact_em(request,contact):
    user_details=User.objects.get(ucontact=contact)
    #print(user_details)
    context={"users":user_details}
    return render(request,"registerform/editcontact_em.html",context)

'''==============================================================EDIT CONTACT - EMPLOYEE==============================================================================================='''
def editvalue_contact_em(request,contact):
    if request.method=="POST":
        new_contact=request.POST.get("editedcontact")
        #print(new_name)
        if new_contact==contact:
            messages.error(request,"Same as old contact")
        elif User.objects.filter(ucontact=new_contact).exists():
            messages.error(request,"Contact already exists")
        else:
            users=User.objects.get(ucontact=contact)
            #print(users)
            users.ucontact=new_contact
            users.save()
            markatts=Markatt.objects.filter(contact=contact)
            for markatt in markatts:
                markatt.contact=new_contact
                markatt.save()
            context={"users":markatts}
            messages.success(request,"Edited successfully")
            return render(request,"registerform/userin.html",context)
    markatts=Markatt.objects.filter(contact=contact)
    context={"users":markatts}
    return render(request,"registerform/userin.html",context)
'''=============================================================EDIT NAME -EMpLOYEE page====================================================================================='''
def editname_em(request,name):
    user_details=User.objects.get(uname=name)
    #print(user_details)
    context={"users":user_details}
    return render(request,"registerform/editname_em.html",context)
'''=============================================================EDIT NAME -EMPLOYEE======================================================================================'''
def editvalue_em(request,name):
    if request.method=="POST":
        new_name=request.POST.get("editedname")
        #print(new_name)
        if new_name==name:
            messages.error(request,"Same as old name")
        elif User.objects.filter(uname=new_name).exists():
            messages.error(request,"Username already exists")
        else:
            users=User.objects.get(uname=name)
            #print(users)
            users.uname=new_name
            users.save()
            markatts=Markatt.objects.filter(name=name)
            for markatt in markatts:
                markatt.name=new_name
                markatt.save()
            context={"users":markatts}
            messages.success(request,"Edited successfully")
            return render(request,"registerform/userin.html",context)
    markatts=Markatt.objects.filter(name=name)
    context={"users":markatts}
    return render(request,"registerform/userin.html",context)
'''===============================================================PREPROCESS AND PREDICTION OF IMAGE======================================================================='''
model_path="c:/facerecog/facerecognition/registerform/ml/recognition_cnn.keras"
model=load_model(model_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image from {img_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        (x_coord, y_coord, w, h) = faces[0]
        face_img = img[y_coord:y_coord + h, x_coord:x_coord + w]
        resi = cv2.resize(face_img, (224, 224))
        resize = cv2.cvtColor(resi, cv2.COLOR_BGR2RGB)
        resi_scaled = resize / 255.0
        resi_scaled = np.expand_dims(resi_scaled, axis=0)
        prediction = model.predict(resi_scaled)
        return np.argmax(prediction)
    else:
        print(f"No face detected in image {img_path}")
        return None

'''=============================================================Mark Attendance=========================================================================================='''
def mark_attendence(user):
    today = datetime.date.today()
    attendance_count = Markatt.objects.filter(userid=user.uid, date=today).count()
    if attendance_count % 2 == 0:
        status = "checkin"
    else:
        status = "checkout"
    attendance=Markatt(userid=user.uid,name=user.uname,email=user.uemail,contact=user.ucontact,time=datetime.datetime.now().time(),date=datetime.date.today(),status=status)
    print(datetime.date.today())
    print(user.uname)
    attendance.save()
'''===============================================================Training data ================================================================================='''
'''
def attendance(request):
    captured_count = 0
    capture_count = 5  # Capture 5 images

    def capture_image():
        nonlocal captured_count
        nonlocal cap
        
        if captured_count < capture_count:
            ret, frame = cap.read()
            if ret:
                user_directory = os.path.join(settings.TEMP_ROOT, 'user_attendance')
                os.makedirs(user_directory, exist_ok=True)  # Create directory if it doesn't exist
                img_name = f'user_attendance_{captured_count}.jpg'
                img_path = os.path.join(user_directory, img_name)
                
                cv2.imwrite(img_path, frame)  # Save the image

                captured_count += 1  # Increment captured image count
                root.after(1000, capture_image)  # Capture image every 1 second
            else:
                messagebox.showerror("Error", "Failed to capture image")

    def update_frame():
        nonlocal cap
        
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
        else:
            messagebox.showerror("Error", "Failed to capture image")

        # Schedule the next update after 10 ms
        lmain.after(10, update_frame)

    root = tk.Tk()
    root.title("Capture Image")
    root.attributes("-topmost", True)
    
    lmain = tk.Label(root)
    lmain.pack()
    
    label = tk.Label(root, text="Capturing images... Please stay in front of the camera.")
    label.pack(pady=5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access camera")
        root.destroy() # Return an empty list or handle appropriately

    update_frame()
    root.after(1000, capture_image)  # Start capturing images after 1 second
    root.mainloop()

    cap.release()
    return render(request,'registerform/home.html',{})
'''
def attendance(request):
    def on_ok_button_click():
        ret, frame = cap.read()
        if ret:
            ret, buf = cv2.imencode(".jpg", frame)
            if ret:
                user_directory = os.path.join(settings.TEMP_ROOT, 'user_attendance')
                os.makedirs(user_directory, exist_ok=True)  # Create directory if it doesn't exist
                img_name = 'user_attendance.jpg'
                img_path = os.path.join(user_directory, img_name)
                with open(img_path, 'wb') as f:
                    f.write(buf.tobytes())
                root.destroy()
                imagepred=predict_image(img_path)
                if imagepred is not None:
                    messagebox.showinfo("Prediction Result", f"Prediction: {imagepred}")
                    # Fetch user based on prediction (order)
                    users = User.objects.all()
                    if imagepred >= 0 and imagepred < len(users):
                        imagepred=int(imagepred)
                        user=users[imagepred]
                        mark_attendence(user)
                    else:
                        messagebox.showerror("Error", "Invalid prediction index")
        else:
            messagebox.showerror("Error", "Failed to capture image")

    def update_frame():
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            root.after(10, update_frame)
        else:
            messagebox.showerror("Error", "Failed to capture image")

    root = tk.Tk()
    root.attributes("-topmost",True)
    root.title("Capture Image")

    lmain = tk.Label(root)
    lmain.pack()
    label = tk.Label(root, text="Press OK to capture the image")
    label.pack(pady=5)

    ok_button = tk.Button(root, text="OK", command=on_ok_button_click)
    ok_button.pack(pady=5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access camera")
        return None

    update_frame()
    root.lift()
    root.mainloop()

    cap.release()

    return render(request,'registerform/home.html',{})

