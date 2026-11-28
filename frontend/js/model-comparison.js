// The research page uses the original Kaggle-exported PNG charts and fixed experiment results.
// No API request or chart recreation is required on this page.
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.visual-card img').forEach((image) => {
    image.addEventListener('error', () => {
      const figure = image.closest('.visual-card');
      if (figure) {
        figure.classList.add('image-missing');
        image.alt = 'Visualization image could not be loaded. Check the reports folder path.';
      }
    }, { once: true });
  });
});
