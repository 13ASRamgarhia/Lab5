from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extract features from request data
    weight = data['weight']
    length1 = data['length1']
    length2 = data['length2']
    length3 = data['length3']
    height = data['height']
    width = data['width']
    
    # Create numpy array for the model
    features = np.array([[weight, length1, length2, length3, height, width]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Convert prediction to species name (if necessary)
    # For example, if prediction returns an integer:
    species = ['Species1', 'Species2', 'Species3'][prediction[0]]
    
    return jsonify({'prediction': species})

if __name__ == '__main__':
    app.run(debug=True)
