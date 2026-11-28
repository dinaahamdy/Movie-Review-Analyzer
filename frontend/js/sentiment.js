
// Word2Vec + LinearSVC Sentiment Analysis

async function analyzeSentiment(text) {
    if (!text || text.trim() === '') {
        return {
            label: 'Neutral',
            method: 'Empty Input',
            score: 0
        };
    }

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ review: text })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Prediction request failed');
        }

        return {
            label: data.sentiment,
            method: `Word2Vec + LinearSVC (${Number(data.confidence).toFixed(1)}% Normalized Model Score)`,
            score: Number(data.confidence)
        };

    } catch (error) {
        console.error("Error calling NLP API:", error);

        return {
            label: 'Neutral',
            method: 'API Connection Error',
            score: 0
        };
    }
}
