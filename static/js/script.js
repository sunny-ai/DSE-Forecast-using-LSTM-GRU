function fetchPredictions() {
    const company = document.getElementById('company').value;
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ company: company }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        renderChart('chart7Days', '7 Days Forecast', data['7_days']);
        renderChart('chart15Days', '15 Days Forecast', data['15_days']);
        renderChart('chart30Days', '30 Days Forecast', data['30_days']);
    })
    .catch(error => {
        console.error('Error fetching predictions:', error);
    });
}

function renderChart(canvasId, label, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    if (window.myCharts && window.myCharts[canvasId]) {
        window.myCharts[canvasId].destroy();  // Destroy existing chart if it exists
    }
    window.myCharts = window.myCharts || {};
    window.myCharts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: data.length }, (_, i) => `Day ${i + 1}`),
            datasets: [{
                label: label,
                data: data,
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 2,
                fill: false,
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: false,
                }
            }
        }
    });
}