from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    
    score = analyzer.polarity_scores(text)['compound']
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

def analyze_dataset(df, column_name):
    
    df['Sentiment'] = df[column_name].astype(str).apply(analyze_sentiment_vader)
    return df

def generate_statistics(df):
    
    sentiment_counts = df['Sentiment'].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm")
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Sentiment Distribution", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plot_path = os.path.join('static', 'sentiment_plot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path

app = Flask(__name__, static_folder='static', template_folder='templates')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    df = pd.read_csv(file_path)
    return jsonify({'columns': df.columns.tolist(), 'file_path': file_path, 'success': 'File uploaded successfully. Click Proceed to continue.'})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    file_path = data.get('file_path')
    column_name = data.get('column')
    if not file_path or not column_name:
        return jsonify({'error': 'Missing file path or column name'})
    df = pd.read_csv(file_path)
    df = analyze_dataset(df, column_name)
    sentiment_counts = df['Sentiment'].value_counts().to_dict()
    plot_path = generate_statistics(df)
    result_file = os.path.join(UPLOAD_FOLDER, 'sentiment_analysis_results.csv')
    df.to_csv(result_file, index=False)
    return jsonify({'sentiment_counts': sentiment_counts, 'result_file': result_file, 'plot_path': plot_path})

@app.route('/download', methods=['GET'])
def download_file():
    file_path = os.path.join(UPLOAD_FOLDER, 'sentiment_analysis_results.csv')
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
