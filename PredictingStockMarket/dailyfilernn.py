# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:35:13 2017

@author: vishw
"""
import os
import shutil

stocks = open('D:\\PatternRecognition\\TermProject\\stockmarket.csv')

dateDif = {}
for line in stocks:
    data = line.split(',')
    date = data[0].split('/')
    diff = data[8]
    for i in range(len(date)):
        if len(date[i]) == 1:
            date[i] = '0' + date[i] 
    date = date[2] + date[0] + date[1]
    dateDif[date] = float(diff)

if 1:
    startpath = 'C:\\TermProject\\FinancialDataset'
    daypath = 'C:\\TermProject\\FinancialDatasetDaily'
    spampath = 'C:\\TermProject\\ClassifiedRnn'
    
    for dirName in [spampath]:
        if os.path.isdir(dirName):
            shutil.rmtree(dirName)
        os.makedirs(dirName)
    
    titleString = ''
    for root, dirs, files in os.walk(startpath):
        for afile in files:
            src = root + '\\' + afile
            story = open(src, 'r', encoding="utf-8")
            title = story.readline()[3:-1]
            if len(title) > 1:
                titleString += title + ' '
            story.close()
        ymd = root.split('\\')[-1]
        if ymd in dateDif:
            print(ymd)
            yDif = dateDif[ymd]
            dif = str(yDif)
            titleString = dif + '\t' + titleString + '\n'
            with open(spampath + '\\' + "ALL" + '.txt', 'a', encoding="utf-8") as day:
                day.write(titleString)
            titleString = ''