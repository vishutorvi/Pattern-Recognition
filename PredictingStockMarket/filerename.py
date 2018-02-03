# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:13:51 2017

@author: vishw
"""

import os

startpath = 'D:\\FinancialDataset'

for root, dirs, files in os.walk(startpath):
    for adir in dirs:
        if 'reuters' not in root and 'reuters' not in adir:
            src = root + '\\' + adir
            dest =  root + '\\' + adir.replace('-','')
            os.rename(src,dest)