# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt


# This file is ploting the composition of the training data
start = os.getcwd()


# load ground truth data
path_txt = start + "\\ISM"
train_txt = open(path_txt + "\\train.txt", "r")
ground_truth = train_txt.readlines()
train_txt.close()

normal_data = 0
covid_data = 0
lung_opacity_data = 0
pneumonia_data = 0

l = len(ground_truth)
for line in ground_truth:
    d = line.split(" ")[1]
    if d[0:6] == "Normal":
        normal_data = normal_data + 1
    elif d[0:5] == "COVID":
        covid_data = covid_data + 1
    elif d[0:9] == "pneumonia":
        pneumonia_data = pneumonia_data + 1
    elif d[0:12] == "Lung_Opacity":
        lung_opacity_data = lung_opacity_data + 1  
    else:
        print("empty")
    
plt.pie([normal_data,covid_data,lung_opacity_data,pneumonia_data], 
        labels = ["normal: " + str(normal_data) + "\nprecent: " + str(100*normal_data/l), "covid: " + str(covid_data) + "\nprecent: " + str(100*covid_data/l), "lung_opacity: " + str(lung_opacity_data) + "\nprecent: " + str(100*lung_opacity_data/l), "pneumonia: " + str(pneumonia_data) + "\nprecent: " + str(100*pneumonia_data/l)])
plt.title("data")
plt.show() 