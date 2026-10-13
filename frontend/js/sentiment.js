// js/sentiment.js
// هذا الملف الآن يتصل بموديل الـ Machine Learning (Logistic Regression) في الباك إند

async function analyzeSentiment(text) {
    // التحقق من أن النص غير فارغ لتجنب خطأ 400 من الخادم
    if (!text || text.trim() === '') {
        return { label: 'Neutral', method: 'Empty Input', score: 0 };
    }

    try {
        // إرسال النص إلى الباك إند
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ review: text })
        });

        const data = await response.json();

        // معالجة الأخطاء القادمة من السيرفر (مثل كود 400 أو 500)
        if (!response.ok) {
            throw new Error(data.error || 'حدث خطأ في الخادم');
        }

        // تطبيق المنطق المذكور في وثيقة الـ API لاستخراج التعليق المحايد
        let finalLabel = data.sentiment;
        if (data.confidence < 60) {
            finalLabel = 'Neutral';
        }

        return {
            label: finalLabel,
            // حفظ اسم الموديل ونسبة الثقة لعرضها في تفاصيل الفيلم
            method: `Logistic Regression (${data.confidence.toFixed(1)}% Confidence)`,
            score: data.confidence
        };

    } catch (error) {
        console.error("Error calling NLP API:", error);
        // حالة فشل الاتصال بالسيرفر (Fallback)
        return { label: 'Neutral', method: 'API Connection Error', score: 0 };
    }
}