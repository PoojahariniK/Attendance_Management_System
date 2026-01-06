# Attendance Management System – Facial Recognition

This project is a facial recognition–based attendance management system designed to automate the process of recording attendance using computer vision and deep learning techniques. The system identifies individuals in real time and securely stores attendance records using a Django-based backend.

---

## Project Description

The Attendance Management System leverages facial recognition to eliminate manual attendance processes. Using OpenCV for face detection and a CNN model for face recognition, the system captures facial data, identifies registered users, and marks attendance automatically. The backend is built with Django and MySQL to ensure reliable data handling and system performance.

---

## Key Features

- Real-time face detection and recognition
- Automated attendance marking
- CNN-based facial recognition model
- Secure storage of attendance records
- Backend developed using Django and MySQL
- Scalable and modular project structure

---

## Technology Stack

- Backend Framework: Django (Python)
- Database: MySQL
- Machine Learning: Keras, Scikit-learn
- Computer Vision: OpenCV
- Deep Learning Model: Convolutional Neural Network (CNN)
- Frontend: Django Templates (HTML, CSS)

---

## System Architecture Overview

1. Facial images are captured using a camera device.
2. OpenCV detects faces from the captured frames.
3. A CNN model processes and identifies the detected faces.
4. Recognized users are matched with registered records.
5. Attendance data is stored securely in the MySQL database through Django.

---

## Project Structure

facerecognition/
│
├── facerecognition/ Django project configuration
├── registerform/ User registration and attendance logic
├── ml/ Facial recognition model and training scripts
├── templates/ HTML templates
├── media/ User facial images (ignored from version control)
├── temp/ Temporary image storage
├── manage.py


---

## Security and Data Handling

- Facial images and database files are excluded from version control
- Environment variables are used for sensitive configurations
- The system follows basic security and privacy-aware development practices

---

## Applications

- Educational institutions
- Corporate attendance systems
- Secure authentication and access control
- Smart surveillance and monitoring solutions

---

## Author

Poojaharini K  
