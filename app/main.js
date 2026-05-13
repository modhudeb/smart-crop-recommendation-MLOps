document.addEventListener('DOMContentLoaded', () => {
    // ---------------------------
    // Helpers & DOM references
    // ---------------------------
    const months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ];

    const monthSelects = document.querySelectorAll('.month-select');
    const form = document.getElementById('cropForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultArea = document.getElementById('resultArea');
    const predictionsList = document.getElementById('predictionsList');

    // Populate month selects (avoids repeating HTML)
    monthSelects.forEach(select => {
        months.forEach(month => {
            const option = document.createElement('option');
            option.value = month;
            option.textContent = month.charAt(0).toUpperCase() + month.slice(1);
            select.appendChild(option);
        });
    });

    // sensible defaults to speed testing
    document.getElementById('growth_start').value = 'july';
    document.getElementById('growth_end').value = 'october';
    document.getElementById('harvest_start').value = 'november';
    document.getElementById('harvest_end').value = 'december';

    // ---------------------------
    // UI helpers
    // ---------------------------
    function showLoading(text = "Analyzing Soil & Climate...") {
        submitBtn.textContent = text;
        submitBtn.disabled = true;
        predictionsList.innerHTML = '';
        resultArea.classList.add('hidden');
    }

    function hideLoading() {
        submitBtn.textContent = "Predict Suitable Crops";
        submitBtn.disabled = false;
    }

    function showError(msg) {
        // simple alert for now, but could be replaced with an inline error box
        alert(msg);
    }

    function formatProbability(prob) {
        // prob assumed 0..1
        if (typeof prob !== 'number' || Number.isNaN(prob)) return '0.0%';
        return (prob * 100).toFixed(1) + '%';
    }

    // ---------------------------
    // Display results
    // ---------------------------
    function displayResults(predictions) {
        predictionsList.innerHTML = '';

        if (!Array.isArray(predictions) || predictions.length === 0) {
            predictionsList.innerHTML = '<div class="prediction-card">No suitable crops found.</div>';
            resultArea.classList.remove('hidden');
            resultArea.scrollIntoView({ behavior: 'smooth' });
            return;
        }

        predictions.forEach(pred => {
            const card = document.createElement('div');
            card.className = 'prediction-card';

            const cropName = document.createElement('span');
            cropName.className = 'crop-name';
            cropName.textContent = (pred.crop || 'Unknown').toString();

            const cropProb = document.createElement('span');
            cropProb.className = 'crop-prob';
            cropProb.textContent = `Confidence: ${formatProbability(Number(pred.probability) || 0)}`;

            card.appendChild(cropName);
            card.appendChild(cropProb);

            predictionsList.appendChild(card);
        });

        resultArea.classList.remove('hidden');
        resultArea.scrollIntoView({ behavior: 'smooth' });
    }

    // ---------------------------
    // Small client-side validation
    // ---------------------------
    function validatePayload(payload) {
        const errors = [];

        if (!payload.district) errors.push("District is required.");
        if (!payload.season) errors.push("Season is required.");
        if (!Number.isFinite(payload.area) || payload.area < 0) errors.push("Area must be a non-negative number.");
        if (!payload.transplant_month) errors.push("Transplant month is required.");
        if (!payload.growth_period.includes('to')) errors.push("Growth period must be in format 'start to end'.");
        if (!payload.harvest_period.includes('to')) errors.push("Harvest period must be in format 'start to end'.");

        ['min_temp','max_temp','min_relative_humidity','max_relative_humidity'].forEach(k => {
            if (!Number.isFinite(payload[k])) errors.push(`${k.replace(/_/g,' ')} must be a number.`);
        });

        return errors;
    }

    // ---------------------------
    // Form submit
    // ---------------------------
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        showLoading();

        // Build payload
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

        // Basic client-side validation
        const validationErrors = validatePayload(payload);
        if (validationErrors.length) {
            hideLoading();
            showError(validationErrors.join('\n'));
            return;
        }

        try {
            // same-origin fetch — no hardcoded host, no CORS gymnastics
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            // Handle non-JSON error responses safely
            if (!response.ok) {
                let errDetail = `Request failed: ${response.status} ${response.statusText}`;
                try {
                    const errJson = await response.json();
                    if (errJson && errJson.detail) errDetail = errJson.detail;
                } catch (_) {
                    // ignore JSON parse errors
                }
                throw new Error(errDetail);
            }

            const data = await response.json();

            if (data && data.status === 'success' && Array.isArray(data.predictions)) {
                displayResults(data.predictions);
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
