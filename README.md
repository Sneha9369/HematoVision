# HematoVision: Advanced Blood Cell Classification Using Transfer Learning
HematoVision is a deep learning-based web application that classifies blood cell images into four types of white blood cells: Eosinophil, Lymphocyte, Monocyte, and Neutrophil. 
The system uses Transfer Learning with the MobileNetV2 model to improve accuracy and reduce training time. The application is built using Flask for the backend and HTML/CSS for the frontend. Users can upload blood cell images and get predictions, and the admin can view prediction history.

## Features
- User Registration and Login
- Upload Blood Cell Image
- Blood Cell Classification using Deep Learning
- Prediction Result Display
- Prediction History Storage
- Admin Panel to View History
- Web-based Application

## Technology Stack

### Frontend
- HTML
- CSS
- Bootstrap

### Backend
- Python
- Flask

### Deep Learning
- TensorFlow
- Keras
- MobileNetV2 (Transfer Learning)

### Tools
- VS Code
- GitHub
- Kaggle Dataset

## Project Structure

HematoVision/
│
├── dataset/
│   ├── TRAIN/
│   └── TEST/
│
├── static/
│   ├── uploads/
│   ├── blood_cells.png
│   ├── history.txt
│   ├── users.txt
│
├── templates/
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── result.html
│   ├── history.html
│   ├── admin.html
│
├── app.py
├── train_model.py
├── Blood_Cell.h5
├── class_names.txt
├── requirements.txt
├── README.md

## System Workflow
1. User registers and logs into the system
2. User uploads a blood cell image
3. Image preprocessing is performed
4. Deep learning model predicts the blood cell type
5. Result is displayed to the user
6. Prediction is stored in history
7. Admin can view prediction history

## Model Information
The model is built using Convolutional Neural Network (CNN) and Transfer Learning with MobileNetV2. The model is trained on a blood cell image dataset and saved as Blood_Cell.h5. This trained model is used in the Flask web application to make predictions.

## Dataset
Dataset used: Blood Cell Images Dataset from Kaggle
https://www.kaggle.com/datasets/paultimothymooney/blood-cells

## Future Scope
- Add more blood cell types
- Improve model accuracy
- Deploy on AWS Cloud
- Convert into mobile application
- Use MySQL database instead of text files

## References
1. Kaggle Blood Cell Dataset
2. MobileNetV2 Research Paper
3. TensorFlow Documentation
4. Flask Documentation
5. Keras Transfer Learning Documentation
6. SmartInternz Internship Documentation

## Acknowledgement
This project was developed as part of the SmartInternz AI & Cloud Virtual Internship. The project guidance, documentation structure, and deployment workflow were referred from the SmartInternz Workspace.
