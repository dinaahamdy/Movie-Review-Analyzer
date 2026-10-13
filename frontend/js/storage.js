// Handles saving and retrieving reviews from localStorage
const STORAGE_KEY = 'moviemind_reviews';

function getReviews() {
    const reviewsJSON = localStorage.getItem(STORAGE_KEY);
    return reviewsJSON ? JSON.parse(reviewsJSON) : [];
}

function saveReview(reviewObj) {
    const reviews = getReviews();
    reviews.push(reviewObj);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(reviews));
}

function getReviewsByMovieId(movieId) {
    return getReviews().filter(review => review.movieId === movieId);
}

// Calculate dynamic statistics for a movie
function getMovieStats(movieId) {
    const reviews = getReviewsByMovieId(movieId);
    let totalStars = 0;
    let ratedReviewsCount = 0;

    reviews.forEach(r => {
        if (r.rating && r.rating > 0) {
            totalStars += parseInt(r.rating);
            ratedReviewsCount++;
        }
    });

    const averageRating = ratedReviewsCount > 0 ? (totalStars / ratedReviewsCount).toFixed(1) : 0;
    return {
        count: reviews.length,
        averageRating: averageRating
    };
}

// Initialize some demo data if completely empty
function initDemoReviews() {
    if (getReviews().length === 0) {
        saveReview({
            id: 'rev-demo-1',
            movieId: 'movie-1',
            reviewerName: 'Alice',
            reviewText: 'A magical masterpiece! The story was beautiful and amazing.',
            rating: 5,
            sentimentLabel: 'Positive',
            sentimentMethod: 'Lexicon-based Baseline',
            timestamp: new Date().toISOString()
        });
        saveReview({
            id: 'rev-demo-2',
            movieId: 'movie-1',
            reviewerName: 'Bob',
            reviewText: 'It was okay. The movie lasts two hours.',
            rating: 3,
            sentimentLabel: 'Neutral',
            sentimentMethod: 'Lexicon-based Baseline',
            timestamp: new Date().toISOString()
        });
    }
}
initDemoReviews();