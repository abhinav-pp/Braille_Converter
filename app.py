import os
from flask import Flask, flash, request, redirect, url_for,render_template
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import subprocess

UPLOAD_FOLDER = r'C:\Users\abhin\Desktop\mini_project\Mini_Project\images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = "jfbwiufjbrwibkfhgbg"
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_ahe_and_save(image_name, output_name):
    image = cv2.imread(image_name, cv2  .IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to read image from {image_name}")
        return
    
    # Apply Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(image)
    
    # Save the equalized image
    cv2.imwrite(output_name,equalized_image)
    print(f"Equalized image saved to {output_name}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    res = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        cols = request.form.get("num_columns")
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            apply_ahe_and_save("./images/"+filename,"./images/"+filename)
            print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(cols)
            cmd = f"python3.11.exe ./f3.py {os.path.join(app.config['UPLOAD_FOLDER'], filename)} {cols}"
            print(cmd)
            res = subprocess.check_output(cmd, shell=True, text=True).split(":")[1]
    return render_template("vjec.html",result = res)

if __name__ == '__main__':
    app.run(debug=True)

