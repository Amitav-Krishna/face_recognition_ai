from flask import Flask, render_template, redirect, url_for, flash
import os
from deepface import DeepFace
import numpy as np
from scipy.spatial.distance import cosine
from werkzeug.utils import secure_filename


# Initialise Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = "supersecretkey"

# Create the uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    app.makedirs(app.config['UPLOAD_FOLDER'])


#Load known
