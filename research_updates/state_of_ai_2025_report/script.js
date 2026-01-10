const slides = document.querySelectorAll('.slide');
let current = 0;

function showSlide(n) {
    slides[current].classList.remove('active');
    current = (n + slides.length) % slides.length;
    slides[current].classList.add('active');
    document.getElementById('current').textContent = current + 1;
    document.getElementById('total').textContent = slides.length;
    document.getElementById('progress').style.width = ((current + 1) / slides.length * 100) + '%';
}

function nextSlide() { showSlide(current + 1); }
function prevSlide() { showSlide(current - 1); }

document.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); nextSlide(); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); prevSlide(); }
});

showSlide(0);
