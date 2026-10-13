document.addEventListener('DOMContentLoaded', () => {
    // القائمة العلوية
    const navElement = document.querySelector('nav');
    if(navElement) {
        navElement.innerHTML = `
            <a href="index.html" class="logo"><span>✨</span> MovieMind</a>
            <ul>
                <li><a href="index.html">Home</a></li>
                <li><a href="movies.html">Explore Movies</a></li>
                <li><a href="write-review.html" class="active">Write a Review</a></li>
                <li><a href="sentiment.html">Sentiment Insights</a></li>
                <li><a href="about-nlp.html">About NLP</a></li>
            </ul>
        `;
    }

    const movieSelect = document.getElementById('movieSelect');
    const form = document.getElementById('reviewForm');
    const successMessage = document.getElementById('successMessage');
    const viewReviewBtn = document.getElementById('viewReviewBtn');

    // ملء قائمة الأفلام
    if(typeof movies !== 'undefined' && movieSelect) {
        movies.forEach(m => {
            const option = document.createElement('option');
            option.value = m.id;
            option.textContent = m.title;
            movieSelect.appendChild(option);
        });

        // تحديد الفيلم تلقائياً من الرابط إن وُجد
        const urlParams = new URLSearchParams(window.location.search);
        const preselectedId = urlParams.get('movieId');
        if (preselectedId && movies.some(m => m.id === preselectedId)) {
            movieSelect.value = preselectedId;
        }
    }

    if(form) {
        // لاحظي إضافة كلمة async هنا
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const movieId = movieSelect.value;
            const reviewerName = document.getElementById('reviewerName').value.trim() || 'Anonymous';
            const reviewText = document.getElementById('reviewText').value.trim();
            
            const selectedStar = document.querySelector('input[name="rating"]:checked');
            const rating = selectedStar ? parseInt(selectedStar.value) : null;

            if (!movieId || !reviewText) {
                alert('Please select a movie and write a review.');
                return;
            }

            // 1. تغيير شكل الزر لإظهار حالة التحميل أثناء عمل الـ API
            const submitBtn = form.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerText;
            submitBtn.innerText = 'Analyzing Sentiment... 🪄';
            submitBtn.disabled = true;
            submitBtn.style.opacity = '0.7';

            // 2. إرسال النص للباك إند والانتظار (await) حتى يعود الرد
            const sentimentResult = await analyzeSentiment(reviewText);

            // 3. تجهيز بيانات المراجعة للحفظ
            const newReview = {
                id: 'rev-' + Date.now(),
                movieId: movieId,
                reviewerName: reviewerName,
                reviewText: reviewText,
                rating: rating,
                sentimentLabel: sentimentResult.label,
                sentimentMethod: sentimentResult.method, 
                timestamp: new Date().toISOString()
            };

            // 4. حفظ البيانات
            saveReview(newReview);

            // 5. إعادة الزر والفورم لحالتهم الطبيعية
            submitBtn.innerText = originalBtnText;
            submitBtn.disabled = false;
            submitBtn.style.opacity = '1';
            
            form.reset();
            document.querySelectorAll('input[name="rating"]').forEach(rb => rb.checked = false);
            
            form.style.display = 'none';
            successMessage.style.display = 'block';
            viewReviewBtn.href = `movie-details.html?id=${movieId}`;
        });
    }
});