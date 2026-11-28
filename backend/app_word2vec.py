
import os
import sys
import json

from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# 1. Project paths and Word2Vec inference engine
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from word2vec_inference import predict_sentiment

# ============================================================
# 2. Flask app
# ============================================================

app = Flask(__name__)
CORS(app)

print("Word2Vec + LinearSVC inference engine loaded successfully!")

# ============================================================
# 3. Routes
# ============================================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Movie Review Analyzer API is running!',
        'model': 'Word2Vec + LinearSVC'
    }), 200


@app.route('/comparison-results', methods=['GET'])
def comparison_results():
    results_path = os.path.join(
        project_root,
        'results',
        'comparison_results.json'
    )

    if not os.path.isfile(results_path):
        return jsonify({
            'error': 'Comparison results have not been generated yet.',
            'hint': 'Generate results/comparison_results.json first.'
        }), 404

    try:
        with open(results_path, 'r', encoding='utf-8') as results_file:
            results = json.load(results_file)

        return jsonify(results), 200

    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({
            'error': f'Could not read comparison results: {exc}'
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                'error': 'No JSON data provided'
            }), 400

        review = str(data.get('review') or '').strip()

        if not review:
            return jsonify({
                'error': 'Field "review" is required'
            }), 400

        result = predict_sentiment(review)

        if not result['cleaned_text']:
            return jsonify({
                'error': 'Review is empty after cleaning.'
            }), 400

        return jsonify({
            'sentiment': result['sentiment'],
            'confidence': round(float(result['confidence']), 2),
            'cleaned_text': result['cleaned_text']
        }), 200

    except Exception as exc:
        app.logger.exception("Prediction failed")

        return jsonify({
            'error': f'Internal server error: {str(exc)}'
        }), 500


# ============================================================
# 4. Run server
# ============================================================

if __name__ == '__main__':
    print("\nMovie Review Analyzer API")
    print("Model: Word2Vec + LinearSVC")

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )
