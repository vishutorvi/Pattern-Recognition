# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:55:38 2017

@author: vishw
"""

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))


print()