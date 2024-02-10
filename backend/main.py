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


app = Flask(__name__)


df = pd.read_csv('final.csv')
GOOGLE_API_KEY = "AIzaSyBYtrF_jcKp0uNsx7zH00ZjOyz4bC8SUTY"

genai.configure(api_key=GOOGLE_API_KEY)
with open('similar.pkl', 'rb') as file:
    model = pickle.load(file)

def get_related_products(index, num_products=10):
    if index < 0 or index >= len(df):
        print(f"Invalid index: {index}")
        return None
    sim_scores = list(enumerate(model[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:num_products+1]]
    return df.iloc[sim_indices][['url', 'name', 'price', 'currency', 'description','images', 'Last_Category']]

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
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


@app.route('/description', methods=["POST"])
def desc():
    img = request.files['file']
    img = Image.open(file)

    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(img)

    description = to_markdown(response.text)

    return description

if __name__ == '__main__':
    # app.run(debug=False)
    app.run(host='0.0.0.0', port=3001)