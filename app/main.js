document.addEventListener('DOMContentLoaded', () => {
    const months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ];

    const monthSelects = document.querySelectorAll('.month-select');
    const form = document.getElementById('cropForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultArea = document.getElementById('resultArea');
    const predictionsList = document.getElementById('predictionsList');
    const resultModal = document.getElementById('resultModal');
    const closeResultModal = document.getElementById('closeResultModal');
    const modalPredictionsList = document.getElementById('modalPredictionsList');
    const topCropHero = document.getElementById('topCropHero');

    monthSelects.forEach(select => {
        months.forEach(month => {
            const option = document.createElement('option');
            option.value = month;
            option.textContent = month.charAt(0).toUpperCase() + month.slice(1);
            select.appendChild(option);
        });
    });

    document.getElementById('growth_start').value = 'july';
    document.getElementById('growth_end').value = 'october';
    document.getElementById('harvest_start').value = 'november';
    document.getElementById('harvest_end').value = 'december';

    function showLoading(text = "Analyzing Soil & Climate...") {
        submitBtn.textContent = text;
        submitBtn.disabled = true;
        predictionsList.innerHTML = '';
        resultArea.classList.add('hidden');
        topCropHero.innerHTML = `
            <span class="hero-rank">Scanning field profile</span>
            <strong>Analyzing...</strong>
            <span class="hero-confidence">Preparing ranked crop matches</span>
        `;
        modalPredictionsList.innerHTML = `
            <div class="ranked-crop loading-row">
                <div class="crop-rank">...</div>
                <div class="crop-details">
                    <div class="crop-name">Matching climate, season, and region</div>
                    <div class="crop-confidence">This can take a few seconds for the ensemble model.</div>
                    <div class="confidence-meter"><span style="width: 62%"></span></div>
                </div>
            </div>
        `;
        resultModal.classList.remove('hidden');
        resultModal.setAttribute('aria-hidden', 'false');
    }

    function hideLoading() {
        submitBtn.textContent = "Predict Suitable Crops";
        submitBtn.disabled = false;
    }

    function showError(msg) {
        alert(msg);
    }

    function formatConfidence(value) {
        if (value === null || value === undefined) return 'N/A';
        return `${Math.round(Number(value) * 100)}%`;
    }

    function normalizePredictions(data) {
        if (Array.isArray(data.predictions) && data.predictions.length) {
            return data.predictions;
        }
        if (data.prediction) {
            return [{ rank: 1, crop: data.prediction, confidence: null }];
        }
        return [];
    }

    function renderRankedCrop(prediction) {
        const row = document.createElement('div');
        row.className = 'ranked-crop';

        const rank = document.createElement('div');
        rank.className = 'crop-rank';
        rank.textContent = `#${prediction.rank}`;

        const details = document.createElement('div');
        details.className = 'crop-details';

        const name = document.createElement('div');
        name.className = 'crop-name';
        name.textContent = prediction.crop;

        const confidence = document.createElement('div');
        confidence.className = 'crop-confidence';
        confidence.textContent = `Confidence ${formatConfidence(prediction.confidence)}`;

        const meter = document.createElement('div');
        meter.className = 'confidence-meter';
        const fill = document.createElement('span');
        fill.style.width = prediction.confidence === null || prediction.confidence === undefined
            ? '0%'
            : `${Math.max(3, Math.round(Number(prediction.confidence) * 100))}%`;
        meter.appendChild(fill);

        details.appendChild(name);
        details.appendChild(confidence);
        details.appendChild(meter);
        row.appendChild(rank);
        row.appendChild(details);
        return row;
    }

    function displayResults(data) {
        predictionsList.innerHTML = '';
        modalPredictionsList.innerHTML = '';
        topCropHero.innerHTML = '';

        const predictions = normalizePredictions(data);
        if (!predictions.length) {
            predictionsList.innerHTML = '<div class="prediction-card">No suitable crop found.</div>';
            resultArea.classList.remove('hidden');
            return;
        }

        const top = predictions[0];
        const heroRank = document.createElement('span');
        heroRank.className = 'hero-rank';
        heroRank.textContent = `Rank #${top.rank}`;

        const heroName = document.createElement('strong');
        heroName.textContent = top.crop;

        const heroConfidence = document.createElement('span');
        heroConfidence.className = 'hero-confidence';
        heroConfidence.textContent = `Confidence ${formatConfidence(top.confidence)}`;

        topCropHero.appendChild(heroRank);
        topCropHero.appendChild(heroName);
        topCropHero.appendChild(heroConfidence);

        predictions.forEach(prediction => {
            const compact = document.createElement('div');
            compact.className = 'prediction-card';
            compact.innerHTML = `<span class="crop-name">#${prediction.rank} ${prediction.crop}</span><span class="crop-prob">${formatConfidence(prediction.confidence)}</span>`;
            predictionsList.appendChild(compact);
            modalPredictionsList.appendChild(renderRankedCrop(prediction));
        });

        resultArea.classList.add('hidden');
        resultModal.classList.remove('hidden');
        resultModal.setAttribute('aria-hidden', 'false');
        closeResultModal.focus();
    }

    closeResultModal.addEventListener('click', () => {
        resultModal.classList.add('hidden');
        resultModal.setAttribute('aria-hidden', 'true');
    });

    resultModal.addEventListener('click', (event) => {
        if (event.target === resultModal) {
            resultModal.classList.add('hidden');
            resultModal.setAttribute('aria-hidden', 'true');
        }
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && !resultModal.classList.contains('hidden')) {
            resultModal.classList.add('hidden');
            resultModal.setAttribute('aria-hidden', 'true');
        }
    });

    function validatePayload(payload) {
        const errors = [];

        if (!payload.district) errors.push("District is required.");
        if (!payload.season) errors.push("Season is required.");
        if (!Number.isFinite(payload.area) || payload.area < 0) errors.push("Area must be a non-negative number.");
        if (!payload.transplant_month) errors.push("Transplant month is required.");
        if (!payload.growth_period.includes(' to ')) errors.push("Growth period must be in format 'start to end'.");
        if (!payload.harvest_period.includes(' to ')) errors.push("Harvest period must be in format 'start to end'.");

        ['min_temp','max_temp','min_relative_humidity','max_relative_humidity'].forEach(k => {
            if (!Number.isFinite(payload[k])) errors.push(`${k.replace(/_/g,' ')} must be a number.`);
        });

        return errors;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        showLoading();

        const growthPeriod = `${document.getElementById('growth_start').value} to ${document.getElementById('growth_end').value}`;
        const harvestPeriod = `${document.getElementById('harvest_start').value} to ${document.getElementById('harvest_end').value}`;

        const payload = {
            district: document.getElementById('district').value,
            season: document.getElementById('season').value,
            area: parseFloat(document.getElementById('area').value),
            transplant_month: document.getElementById('transplant_month').value,
            growth_period: growthPeriod,
            harvest_period: harvestPeriod,
            min_temp: parseFloat(document.getElementById('min_temp').value),
            max_temp: parseFloat(document.getElementById('max_temp').value),
            min_relative_humidity: parseFloat(document.getElementById('min_rh').value),
            max_relative_humidity: parseFloat(document.getElementById('max_rh').value)
        };

        const validationErrors = validatePayload(payload);
        if (validationErrors.length) {
            hideLoading();
            showError(validationErrors.join('\n'));
            return;
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                let errDetail = `Request failed: ${response.status} ${response.statusText}`;
                try {
                    const errJson = await response.json();
                    if (errJson && errJson.detail) errDetail = errJson.detail;
                } catch (_) {}
                throw new Error(errDetail);
            }

            const data = await response.json();

            if (data && data.status === 'success' && (data.prediction || data.predictions)) {
                displayResults(data);
            } else {
                throw new Error('Unexpected response format from server.');
            }
        } catch (err) {
            console.error('Prediction error:', err);
            showError(`Prediction error: ${err.message}`);
        } finally {
            hideLoading();
        }
    });
});
