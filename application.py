from flask import Flask,render_template,request,jsonify
import sys
from src.exception import CustomException


@app.route('/')
def home_page():
    return render_template('index.html')
