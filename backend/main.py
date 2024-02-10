from flask import Flask, request, jsonify, send_file
import re
import google.generativeai as genai
import marko
import pickle
import pandas as pd
from flask_cors import CORS
import os
from PIL import Image
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import requests
from io import BytesIO


app = Flask(__name__)


df = pd.read_csv('final.csv')
GOOGLE_API_KEY = "AIzaSyBYtrF_jcKp0uNsx7zH00ZjOyz4bC8SUTY"
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
HEADERS = {"Authorization": "Bearer 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"}
genai.configure(api_key=GOOGLE_API_KEY)
with open('similar.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/query', methods=['POST'])
def query():
    try:
        # Get the JSON data from the request
        json_data = request.get_json()

        # Extract the 'txt' field from the JSON data
        txt = json_data.get('txt')

        # Send the 'txt' data to the Hugging Face model
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": txt})
        image_bytes = response.content

        # Convert the image bytes to PIL Image
        image = Image.open(BytesIO(image_bytes))

        # Save the image to a file (optional)
        image.save('output.jpg')

        # Return the image as a response
        img_io = BytesIO()
        image.save(img_io, 'JPEG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg', as_attachment=True, download_name='output.jpg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_related_products(index, num_products=10):
    if index < 0 or index >= len(df):
        print(f"Invalid index: {index}")
        return None
    sim_scores = list(enumerate(model[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:num_products+1]]
    return df.iloc[sim_indices][['name', 'images']]

@app.route('/recommend', methods=['POST'])
def recommend():
    # Assuming the request contains the product_id for which recommendations are needed
    product_id = request.json['product_id']

    # Get top 5 related products
    recommendations = get_related_products(product_id, num_products=5)

    if recommendations is not None:
        result = recommendations.to_json(orient="records")
        print(result)
        return result
    else:
        return jsonify({'message': 'Product not found'}), 404


@app.route('/market', methods=["POST"])
def market():
    df = pd.read_csv('final.csv')
    selected_col = ['url', 'name', 'price', 'currency', 'description','images', 'Last_Category']
    selected_data = df[selected_col].to_dict(orient="records")
    return jsonify(selected_data)


def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)


@app.route('/description', methods=["POST"])
def desc():
    # Get the image URL from the JSON request
    json_data = request.get_json()
    img_url = json_data.get('url')

    if not img_url:
        return jsonify({'error': 'Image URL not provided'})

    # Download the image from the URL
    response = requests.get(img_url)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to download the image'})

    # Process the downloaded image
    img = Image.open(BytesIO(response.content))
    
    # Assuming 'genai' and 'to_markdown' functions are defined elsewhere
    model = genai.GenerativeModel('gemini-pro-vision')
    prompt = 'Generate a story in 300-400 words based on its history of design and significance. Please do not include "\n" or new line'
    response = model.generate_content([prompt, img])
    description = to_markdown(response.text)
    lines = description.split('\r\n')
    print(description)
    return jsonify({'description': description})

if __name__ == '__main__':
    # app.run(debug=False)
    app.run(host='0.0.0.0', port=3001)