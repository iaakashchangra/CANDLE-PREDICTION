/**
 * Main JavaScript file for Candlestick Pattern Prediction System
 * Handles global functionality, API calls, and UI interactions
 */

// Global variables
let currentUser = null;
let chartInstance = null;
let predictionData = [];
let modelPerformance = {};
let realTimeInterval = null;

// API endpoints
const API_ENDPOINTS = {
    auth: {
        login: '/api/auth/login',
        register: '/api/auth/register',
        logout: '/api/auth/logout',
        profile: '/api/auth/profile'
    },
    data: {
        chart: '/api/chart/data',
        realtime: '/api/data/realtime',
        historical: '/api/data/historical'
    },
    predictions: {
        create: '/api/predictions/create',
        generate: '/api/predictions/generate',
        history: '/api/predictions/history',
        details: '/api/predictions/'
    },
    models: {
        train: '/api/models/train',
        performance: '/api/models/performance',
        compare: '/api/models/compare'
    },
    config: {
        user: '/api/config/user',
        update: '/api/config/update'
    }
};

// Utility functions
class Utils {
    static formatCurrency(value, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    }
    
    static formatNumber(value, decimals = 2) {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(value);
    }
    
    static formatPercentage(value, decimals = 2) {
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(value / 100);
    }
    
    static formatDateTime(dateString) {
        const date = new Date(dateString);
        return date.toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
    
    static formatTimeAgo(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffInSeconds = Math.floor((now - date) / 1000);
        
        if (diffInSeconds < 60) return 'Just now';
        if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
        if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
        return `${Math.floor(diffInSeconds / 86400)}d ago`;
    }
    
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    static showToast(message, type = 'info', duration = 3000) {
        const toastContainer = document.getElementById('toastContainer') || this.createToastContainer();
        const toast = this.createToast(message, type);
        
        toastContainer.appendChild(toast);
        
        // Trigger animation
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Auto remove
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
    
    static createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
        return container;
    }
    
    static createToast(message, type) {
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" onclick="this.parentElement.parentElement.remove()"></button>
            </div>
        `;
        return toast;
    }
}

// API client
class APIClient {
    static async request(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            credentials: 'same-origin'
        };
        
        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            }
            
            return await response.text();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }
    
    static async get(url) {
        return this.request(url, { method: 'GET' });
    }
    
    static async post(url, data) {
        return this.request(url, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }
    
    static async postForm(url, formData) {
        return this.request(url, {
            method: 'POST',
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: formData
        });
    }
    
    static async put(url, data) {
        return this.request(url, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }
    
    static async delete(url) {
        return this.request(url, { method: 'DELETE' });
    }
}

// Chart manager
class ChartManager {
    constructor(containerId) {
        this.containerId = containerId;
        this.chart = null;
        this.data = null;
        this.config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
        };
    }
    
    async loadData(params) {
        try {
            const formData = new FormData();
            Object.keys(params).forEach(key => {
                formData.append(key, params[key]);
            });
            
            const response = await APIClient.postForm(API_ENDPOINTS.data.chart, formData);
            
            if (response.success) {
                this.data = response.chart_data;
                return response;
            } else {
                // Handle specific error messages
                let errorMessage = response.error || 'Failed to load chart data';
                let errorDetails = '';
                
                // Check for specific error types
                if (response.details && Array.isArray(response.details) && response.details.length > 0) {
                    errorDetails = response.details[0];
                }
                
                // Format user-friendly error message
                if (errorMessage.includes('not found') || errorMessage.includes('invalid')) {
                    errorMessage = `Symbol "${params.symbol}" not found or invalid. Please check the company symbol.`;
                } else if (errorMessage.includes('rate limit') || errorMessage.includes('Too Many Requests')) {
                    errorMessage = 'API rate limits exceeded. Please try again in a few minutes.';
                } else if (errorMessage.includes('insufficient data')) {
                    errorMessage = `Insufficient historical data available for ${params.symbol}. Try a different symbol or timeframe.`;
                }
                
                throw new Error(errorMessage);
            }
        } catch (error) {
            console.error('Error loading chart data:', error);
            // Show a more user-friendly error message
            Utils.showToast(error.message, 'danger', 5000);
            
            // Display a message in the chart container
            const container = document.getElementById(this.containerId);
            if (container) {
                container.innerHTML = `
                    <div class="alert alert-danger text-center p-5 m-3">
                        <h4><i class="fas fa-exclamation-triangle me-2"></i>Data Error</h4>
                        <p>${error.message}</p>
                        <small>Try a different symbol or timeframe</small>
                    </div>
                `;
            }
            
            throw error;
        }
    }
    
    render(data = null) {
        if (data) this.data = data;
        if (!this.data) return;
        
        const traces = this.createTraces();
        const layout = this.createLayout();
        
        if (this.chart) {
            Plotly.react(this.containerId, traces, layout, this.config);
        } else {
            Plotly.newPlot(this.containerId, traces, layout, this.config);
            this.chart = true;
        }
    }
    
    createTraces() {
        const traces = [];
        
        // Candlestick trace
        if (this.data.timestamps && this.data.open) {
            traces.push({
                x: this.data.timestamps,
                open: this.data.open,
                high: this.data.high,
                low: this.data.low,
                close: this.data.close,
                type: 'candlestick',
                name: 'Price',
                increasing: { line: { color: '#00ff00' } },
                decreasing: { line: { color: '#ff0000' } }
            });
        }
        
        // Volume trace
        if (this.data.volume) {
            traces.push({
                x: this.data.timestamps,
                y: this.data.volume,
                type: 'bar',
                name: 'Volume',
                yaxis: 'y2',
                opacity: 0.3,
                marker: { color: 'blue' }
            });
        }
        
        // Prediction traces
        if (this.data.predictions) {
            traces.push({
                x: this.data.predictions.map(p => p.timestamp),
                y: this.data.predictions.map(p => p.predicted_close),
                mode: 'markers+lines',
                type: 'scatter',
                name: 'Predictions',
                line: { color: 'orange', dash: 'dash' },
                marker: { color: 'orange', size: 8 }
            });
        }
        
        return traces;
    }
    
    createLayout() {
        return {
            title: {
                text: this.data.title || 'Candlestick Chart',
                font: { size: 16 }
            },
            xaxis: {
                title: 'Time',
                rangeslider: { visible: false }
            },
            yaxis: {
                title: 'Price ($)',
                side: 'left'
            },
            yaxis2: {
                title: 'Volume',
                overlaying: 'y',
                side: 'right',
                showgrid: false
            },
            showlegend: true,
            margin: { l: 50, r: 50, t: 50, b: 50 },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        };
    }
    
    addIndicator(type, data) {
        // Add technical indicators to the chart
        const trace = this.createIndicatorTrace(type, data);
        if (trace) {
            Plotly.addTraces(this.containerId, trace);
        }
    }
    
    createIndicatorTrace(type, data) {
        switch (type) {
            case 'sma':
                return {
                    x: data.timestamps,
                    y: data.values,
                    mode: 'lines',
                    type: 'scatter',
                    name: `SMA(${data.period})`,
                    line: { color: 'blue', width: 2 }
                };
            case 'ema':
                return {
                    x: data.timestamps,
                    y: data.values,
                    mode: 'lines',
                    type: 'scatter',
                    name: `EMA(${data.period})`,
                    line: { color: 'red', width: 2 }
                };
            default:
                return null;
        }
    }
    
    exportChart(format = 'png', filename = 'chart') {
        Plotly.downloadImage(this.containerId, {
            format: format,
            width: 1200,
            height: 800,
            filename: filename
        });
    }
}

// Prediction manager
class PredictionManager {
    static async generatePrediction(params) {
        try {
            const formData = new FormData();
            Object.keys(params).forEach(key => {
                formData.append(key, params[key]);
            });
            
            const response = await APIClient.postForm(API_ENDPOINTS.predictions.generate, formData);
            
            if (response.success) {
                Utils.showToast('Prediction generated successfully', 'success');
                return response;
            } else {
                // Handle specific error messages
                let errorMessage = response.error || 'Failed to generate prediction';
                
                // Format user-friendly error message
                if (errorMessage.includes('not found') || errorMessage.includes('invalid symbol')) {
                    errorMessage = `Symbol "${params.symbol}" not found or invalid. Please check the company symbol.`;
                } else if (errorMessage.includes('rate limit') || errorMessage.includes('Too Many Requests')) {
                    errorMessage = 'API rate limits exceeded. Please try again in a few minutes.';
                } else if (errorMessage.includes('insufficient data')) {
                    errorMessage = `Insufficient historical data available for ${params.symbol}. Try a different symbol or timeframe.`;
                }
                
                throw new Error(errorMessage);
            }
        } catch (error) {
            console.error('Error generating prediction:', error);
            
            // Show a more descriptive error message
            Utils.showToast(error.message, 'danger', 5000);
            
            // If there's a prediction container, show the error there too
            const predictionContainer = document.getElementById('predictionResults');
            if (predictionContainer) {
                predictionContainer.innerHTML = `
                    <div class="alert alert-danger text-center p-4 m-3">
                        <h4><i class="fas fa-exclamation-triangle me-2"></i>Prediction Error</h4>
                        <p>${error.message}</p>
                        <small>Try a different symbol, timeframe, or prediction parameters</small>
                    </div>
                `;
            }
            
            throw error;
        }
    }
    
    static async getPredictionHistory(userId = null) {
        try {
            const url = userId ? `${API_ENDPOINTS.predictions.history}?user_id=${userId}` : API_ENDPOINTS.predictions.history;
            const response = await APIClient.get(url);
            
            if (response.success) {
                return response.predictions;
            } else {
                throw new Error(response.error || 'Failed to load prediction history');
            }
        } catch (error) {
            console.error('Error loading prediction history:', error);
            Utils.showToast('Failed to load prediction history', 'danger');
            throw error;
        }
    }
    
    static async getPredictionDetails(predictionId) {
        try {
            const response = await APIClient.get(`${API_ENDPOINTS.predictions.details}${predictionId}`);
            
            if (response.success) {
                return response.prediction;
            } else {
                throw new Error(response.error || 'Failed to load prediction details');
            }
        } catch (error) {
            console.error('Error loading prediction details:', error);
            Utils.showToast('Failed to load prediction details', 'danger');
            throw error;
        }
    }
}

// Model manager
class ModelManager {
    static async trainModel(params) {
        try {
            const response = await APIClient.post(API_ENDPOINTS.models.train, params);
            
            if (response.success) {
                Utils.showToast('Model training started', 'info');
                return response;
            } else {
                throw new Error(response.error || 'Failed to start model training');
            }
        } catch (error) {
            console.error('Error training model:', error);
            Utils.showToast('Failed to start model training', 'danger');
            throw error;
        }
    }
    
    static async getModelPerformance(userId = null) {
        try {
            const url = userId ? `${API_ENDPOINTS.models.performance}?user_id=${userId}` : API_ENDPOINTS.models.performance;
            const response = await APIClient.get(url);
            
            if (response.success) {
                return response.performance;
            } else {
                throw new Error(response.error || 'Failed to load model performance');
            }
        } catch (error) {
            console.error('Error loading model performance:', error);
            Utils.showToast('Failed to load model performance', 'danger');
            throw error;
        }
    }
    
    static async compareModels(modelIds) {
        try {
            const response = await APIClient.post(API_ENDPOINTS.models.compare, { model_ids: modelIds });
            
            if (response.success) {
                return response.comparison;
            } else {
                throw new Error(response.error || 'Failed to compare models');
            }
        } catch (error) {
            console.error('Error comparing models:', error);
            Utils.showToast('Failed to compare models', 'danger');
            throw error;
        }
    }
}

// Real-time data manager
class RealTimeManager {
    static start(symbol, callback, interval = 30000) {
        this.stop(); // Stop any existing interval
        
        realTimeInterval = setInterval(async () => {
            try {
                const response = await APIClient.get(`${API_ENDPOINTS.data.realtime}?symbol=${symbol}`);
                if (response.success && callback) {
                    callback(response.data);
                }
            } catch (error) {
                console.error('Error fetching real-time data:', error);
            }
        }, interval);
    }
    
    static stop() {
        if (realTimeInterval) {
            clearInterval(realTimeInterval);
            realTimeInterval = null;
        }
    }
}

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Add loading states to forms
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            }
        });
    });
    
    // Add smooth scrolling to anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card, .metric-card, .prediction-card, .model-card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    });
    
    cards.forEach(card => {
        observer.observe(card);
    });
});

// Export global objects
window.Utils = Utils;
window.APIClient = APIClient;
window.ChartManager = ChartManager;
window.PredictionManager = PredictionManager;
window.ModelManager = ModelManager;
window.RealTimeManager = RealTimeManager;