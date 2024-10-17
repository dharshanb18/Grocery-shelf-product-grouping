from flask import Flask, request, render_template, redirect, url_for, session, send_file
from detection_module import predict_products
from grouping_module import get_grouped_labels
from annotated_image import process_image_and_save_annotated

import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'shelf images detection/uploads/'
ANNOTATED_FOLDER = 'shelf images detection/annotated/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def process_file():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        session['image_path'] = image_path

        detected_products = predict_products(image_path)

        session['detected_products'] = detected_products

        return redirect(url_for('show_predictions'))

@app.route('/predictions')
def show_predictions():
    detected_products = session.get('detected_products', [])
    if not detected_products:
        return redirect(url_for('upload_file'))

    return render_template('predictions.html', products=detected_products)

@app.route('/grouping')
def show_grouping():
    detected_products = session.get('detected_products', [])
    if not detected_products:
        return redirect(url_for('upload_file'))

    labels = [product['cnn_class_name'] for product in detected_products]  # Extract labels
    grouped_labels = get_grouped_labels(labels)

    # Store grouped labels in session
    session['grouped_labels'] = grouped_labels

    return render_template('grouping.html', grouped_labels=grouped_labels)

@app.route('/annotated')
def show_annotated_image():
    image_path = session.get('image_path')
    if not image_path:
        return redirect(url_for('upload_file'))

    #Annotate the image
    annotated_image_filename = process_image_and_save_annotated(image_path)  # Get filename of the annotated image
    tmp = os.path.join(r"shelf images detection\annotated",annotated_image_filename)
    return render_template('annotated.html', annotated_image_filename=tmp)


@app.route('/annotated/<filename>')
def display_image(filename):
    return send_file(os.path.join(app.config['ANNOTATED_FOLDER'], filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
