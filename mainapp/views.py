from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
# Create your views here.
def homepage(request):
    return render(request,'homepage.html')
def results(request):
    return render(request,'results.html')
    
def upload(request):
    if request.method == 'POST':
        filename,ext=str(request.FILES['file']).split('.')
        if(ext in ['png','jpg','jpeg','img']):
            filename=request.POST.get("name")+'.'+ext
            handle_uploaded_file(request.FILES['file'], filename,ext)
            return render(request,'results.html')

        else:
            return HttpResponse("<center><h1>Only image files are allowed</h1></center>")


    return HttpResponse("Failed")

def handle_uploaded_file(file, filename,ext):
    with open('test.'+ext, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    classify_face("test.jpg")
    if not os.path.exists('dataset/'):
        os.mkdir('dataset/')
    if not os.path.exists('dataset/'+filename):
        with open('dataset/' + filename, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
    

def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./dataset"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("dataset/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("dataset/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
    #imS = cv2.resize(img, (500,500)) 
    cv2.imwrite('staticfiles/result.png',img)
        