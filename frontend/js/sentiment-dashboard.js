document.addEventListener('DOMContentLoaded', () => {
    // Nav injection
    document.querySelector('nav').innerHTML = `
        <a href="index.html" class="logo"><span>✨</span> MovieMind</a>
        <ul>
            <li><a href="index.html">Home</a></li>
            <li><a href="movies.html">Explore Movies</a></li>
            <li><a href="write-review.html">Write a Review</a></li>
            <li><a href="sentiment.html" class="active">Sentiment Insights</a></li>
            <li><a href="about-nlp.html">About NLP</a></li>
        </ul>
    `;

    const reviews = getReviews();
    if(reviews.length === 0) {
        document.getElementById('dashboardContent').innerHTML = `
            <div class="empty-state">
                <h2>No data to analyze.</h2>
                <p>Submit reviews to see the sentiment dashboard come to life!</p>
                <a href="write-review.html" class="btn" style="margin-top:1rem;">Write a Review</a>
            </div>`;
        return;
    }

    let pos = 0, neu = 0, neg = 0;
    const movieSentimentCount = {}; 
    movies.forEach(m => { movieSentimentCount[m.id] = { title: m.title, pos:0, neu:0, neg:0 }; });

    reviews.forEach(r => {
        if(r.sentimentLabel === 'Positive') { pos++; if(movieSentimentCount[r.movieId]) movieSentimentCount[r.movieId].pos++; }
        else if(r.sentimentLabel === 'Negative') { neg++; if(movieSentimentCount[r.movieId]) movieSentimentCount[r.movieId].neg++; }
        else { neu++; if(movieSentimentCount[r.movieId]) movieSentimentCount[r.movieId].neu++; }
    });

    document.getElementById('totalVal').innerText = reviews.length;
    document.getElementById('posVal').innerText = pos;
    document.getElementById('neuVal').innerText = neu;
    document.getElementById('negVal').innerText = neg;

    // Charts Configuration
    const doughnutCtx = document.getElementById('doughnutChart').getContext('2d');
    new Chart(doughnutCtx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                data: [pos, neu, neg],
                backgroundColor: ['#28a745', '#6c757d', '#dc3545'],
                borderWidth: 0
            }]
        },
        options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
    });

    const activeMovies = Object.values(movieSentimentCount).filter(m => m.pos > 0 || m.neu > 0 || m.neg > 0);
    const labels = activeMovies.map(m => m.title.length > 15 ? m.title.substring(0,15)+'...' : m.title);
    
    const barCtx = document.getElementById('barChart').getContext('2d');
    new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: 'Positive', data: activeMovies.map(m=>m.pos), backgroundColor: '#28a745' },
                { label: 'Neutral', data: activeMovies.map(m=>m.neu), backgroundColor: '#6c757d' },
                { label: 'Negative', data: activeMovies.map(m=>m.neg), backgroundColor: '#dc3545' }
            ]
        },
        options: { responsive: true, scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } } }
    });
});