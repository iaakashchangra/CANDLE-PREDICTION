class MultiUserDashboard {
    constructor() {
        this.charts = new Map();
        this.selection = null;
        this.updateInterval = 5000; // 5 seconds
    }
    
    async initialize() {
        // Get selection data
        const response = await fetch('/api/selections/current');
        this.selection = await response.json();
        
        // Create charts container
        const container = document.getElementById('charts-container');
        container.innerHTML = '';
        
        // Create chart for each user
        for (const user of this.selection.selected_users) {
            const userContainer = document.createElement('div');
            userContainer.className = 'user-chart-container';
            userContainer.innerHTML = `
                <h3>${user.username}</h3>
                <div class="chart-controls">
                    <select class="symbol-select" data-user="${user.user_id}">
                        ${user.symbols.map(s => `<option value="${s}">${s}</option>`).join('')}
                    </select>
                    <select class="timeframe-select" data-user="${user.user_id}">
                        ${user.timeframes.map(t => `<option value="${t}">${t}</option>`).join('')}
                    </select>
                    <select class="model-select" data-user="${user.user_id}">
                        ${user.models.map(m => `<option value="${m}">${m}</option>`).join('')}
                    </select>
                </div>
                <div id="chart-${user.user_id}" class="chart"></div>
            `;
            container.appendChild(userContainer);
            
            // Initialize chart
            this.initializeChart(user);
            
            // Add event listeners
            userContainer.querySelector('.symbol-select').addEventListener('change', 
                () => this.updateChart(user.user_id));
            userContainer.querySelector('.timeframe-select').addEventListener('change', 
                () => this.updateChart(user.user_id));
            userContainer.querySelector('.model-select').addEventListener('change', 
                () => this.updateChart(user.user_id));
        }
        
        // Start auto-update
        setInterval(() => this.updateAllCharts(), this.updateInterval);
    }
    
    async initializeChart(user) {
        const chartElement = document.getElementById(`chart-${user.user_id}`);
        const chart = new CandlestickChart(chartElement);
        this.charts.set(user.user_id, chart);
        await this.updateChart(user.user_id);
    }
    
    async updateChart(userId) {
        const chart = this.charts.get(userId);
        const container = document.querySelector(`#chart-${userId}`).parentElement;
        
        const symbol = container.querySelector('.symbol-select').value;
        const timeframe = container.querySelector('.timeframe-select').value;
        const model = container.querySelector('.model-select').value;
        
        const response = await fetch('/api/chart/candlestick-with-predictions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol,
                timeframe,
                model_type: model,
                prediction_count: this.selection.selected_users
                    .find(u => u.user_id === userId).prediction_count
            })
        });
        
        const data = await response.json();
        if (data.success) {
            chart.updateData(data.chart_data);
        }
    }
    
    async updateAllCharts() {
        for (const userId of this.charts.keys()) {
            await this.updateChart(userId);
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new MultiUserDashboard();
    dashboard.initialize();
});