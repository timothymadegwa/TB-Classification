# TB-Classification [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![contributions welcome](https://img.shields.io/static/v1.svg?label=Contributions&message=Welcome&color=0059b3&style=flat-square)](https://github.com/aqila-ai/pesa-mtaani/develop/README.md)&nbsp; [![Tested on Python 3.8](https://img.shields.io/badge/Tested%20-Python%203.7-blue.svg?logo=python&style=flat-square)]( https://www.python.org/downloads) &nbsp;

### PROJECT OVERVIEW  
The TB-Classification application is an Tuberculosis X-Ray scanning application that returns the probability of a scan (X-Ray) having Tuberculosis.
The model was trained using Keras (VGG16 pre-trained model) and obtained an accuracy of about 92% on the test set (during training)
The Project is not currently hosted but here is a short video on it functionality on YouTube (https://youtu.be/-ax5aX80li0)


 
### Tech/Framework
> Python 3.7.7, Flask 1.1.2   
### Getting started 
> git clone the repo  
> python3 -m venv ./venv  (Create venv)  
> pip install -r requirements.txt (Install python packages)  
> cd deplpyment
> python prediction_model.py

### DATA
The data used to train the model was obtained from Zindi (https://zindi.africa/competitions/runmila-ai-institute-minohealth-ai-labs-tuberculosis-classification-via-x-rays-challenge/data)

### DETAILS
#### Description
It takes a very long time for medical images to be analysed by a radiologist/doctor and during this waiting period, a patient's condition may become worse. This solution aims to reduce the amount of time it takes to scan a medical image hence cutting down on the number of unecessary medical complications 

#### Stack
This is a full stack project with a backend (Python - Flask and Tensorflow (Keras) ) and frontend (HTML, CSS and Bootstrap)

#### Reasons for stack choices
The backend was built on python Flask due to its simple nature (only one page to be rendered) and HTML, CSS and Bootstrap to help bring the solution to be used by non-tech persons by builting simple and easy to use UI for the project

#### Trade-offs
The major trade-off for this project is that the VGG16 model is quite heavy. Should I have more time on it, I would like to explore the use of other pre-trained models such as Mobilenet

#### Personal Profile
I have an online resume (https://timdev.co.ke/) containing links to works I have completed before (code cannot be shared) under the projects page

#### Other projects
1. Kopapap (https://kopapap-api.herokuapp.com/). This is a POC for the use of machine learning as an extra layer of screening before a loan is issues. It is built using Flask, HTML, CSS and Bootstrap
2. S-BOB (https://sbob.aqila.co.ke/). An application used to digitize credit union operations. It allows peer-to-peer payment, loan application and disbursment, airtime purchase among others. It is bult on Django, HTML, CSS, Bootstrap and JavaScript. Test login details are available on my public resume
3. Aqila Inventory (https://inventory.aqila.co.ke/). An application to help small businesses manage thier inventory, make sales and see thier profits. It is built on Django, HTML, CSS, Bootstrap and JavaScript. Test login details are available on my public resume
