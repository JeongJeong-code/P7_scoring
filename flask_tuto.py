# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 18:06:45 2022

@author: nwenz
"""

from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"