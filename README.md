# Enhanced Treasury Yield Analysis System

A comprehensive, production-ready system for analyzing Treasury yields with advanced predictive modeling, multi-source data integration, and real-time alerting capabilities.

## Key Features

### 1. **Predictive Modeling Layer**
- **Vector Autoregressive (VAR) Models**: Advanced time series forecasting for yield curve dynamics
- **Rolling Walk-Forward Validation**: Robust model performance assessment with out-of-sample testing
- **Confidence Intervals**: Statistical uncertainty quantification with Monte Carlo simulation
- **Real-time Forecast Visualization**: Interactive charts with forecast bands and historical comparison

### 2. **Enhanced Data Domain**
- **Multi-Source Integration**: Treasury yields, FRED macroeconomic indicators, GCF repo rates, CME volatility surfaces
- **Asynchronous Data Fetching**: High-performance data retrieval with parallel processing
- **Redis Caching**: Intelligent caching system reducing API calls and improving response times
- **Real-time Market Data**: Integration with yfinance for additional market context

### 3. **Production-Ready Architecture**
- **Containerization**: Docker and Docker Compose for reproducible deployments
- **CI/CD Pipeline**: Automated testing, building, and deployment with GitHub Actions
- **Performance Optimization**: Sub-2-second page refresh latency through caching and async processing
- **Accessibility Standards**: WCAG-compliant UI with keyboard navigation and screen reader support

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   Flask API     │    │   Redis Cache   │
│   (Bootstrap)   │◄──►│   (Python)      │◄──►│   (In-Memory)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   VAR Models    │    │   Data Sources  │    │   Alert System  │
│   (Statsmodels) │    │   (FRED, CME)   │    │   (Real-time)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Installation & Setup

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Redis (optional, for caching)
- FRED API key (optional, for enhanced data)

### Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yield-analysis-system
   ```

2. **Set environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the application**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Main Dashboard: http://localhost:5000
   - Predictive Dashboard: http://localhost:5000/predictive
   - API Documentation: http://localhost:5000/api/health

### Local Development Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Redis (optional)**
   ```bash
   docker run -d -p 6379:6379 redis:7-alpine
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

## Usage Guide

### Basic Dashboard
- **Real-time Yields**: Current Treasury yield curves with historical context
- **Spread Analysis**: Key yield spreads (2s10s, 5s10s, 10s30s) with trend analysis
- **Cycle Analysis**: Pre-defined Fed cutting cycle analysis with regime detection

### Predictive Dashboard
- **Forecast Generation**: Generate yield forecasts with configurable horizons and confidence levels
- **Model Validation**: Walk-forward validation results with performance metrics
- **Real-time Alerts**: Automated alerts for yield curve deviations from forecasts
- **Enhanced Data**: Multi-source data integration with cache status monitoring

## Data Sources

### Primary Sources
- **U.S. Treasury**: Daily yield curve data via Fiscal Data API
- **Federal Reserve (FRED)**: Macroeconomic indicators (CPI, GDP, employment)
- **GCF Repo Rates**: Federal Reserve repo reference rates
- **CME Group**: Treasury options volatility surfaces
- **Yahoo Finance**: Additional market data (ETFs, indices)

### Data Processing
- **Stationarity Testing**: Automatic differencing for VAR modeling
- **Missing Value Handling**: Intelligent interpolation and forward-filling
- **Outlier Detection**: Statistical outlier identification and handling
- **Data Validation**: Schema validation and quality checks

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest tests/test_predictor.py
pytest tests/test_data_fetcher.py
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Performance Tests**: Load and stress testing
- **Accessibility Tests**: WCAG compliance verification

## Deployment

### Cloud Deployment
The system is designed for deployment on:
- **AWS ECS/Fargate**: Container orchestration
- **Google Cloud Run**: Serverless container deployment
- **Azure Container Instances**: Managed container service
- **Kubernetes**: Production orchestration

### CI/CD Pipeline
Automated pipeline includes:
- **Code Quality**: Linting, formatting, and security scanning
- **Testing**: Automated test execution with coverage reporting
- **Building**: Docker image building and optimization
- **Deployment**: Staging and production deployment automation

## Performance Metrics

### Response Times
- **Page Load**: < 2 seconds (with caching)
- **API Response**: < 500ms (cached data)
- **Forecast Generation**: < 30 seconds (VAR model training)
- **Data Refresh**: < 10 seconds (async processing)

### Scalability
- **Concurrent Users**: 100+ simultaneous users
- **Data Volume**: 10+ years of historical data
- **Cache Hit Rate**: > 90% (with Redis)
- **Uptime**: 99.9% availability target

## Security Features

- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: API rate limiting to prevent abuse
- **CORS Configuration**: Proper cross-origin resource sharing
- **Container Security**: Non-root user and minimal attack surface
- **Dependency Scanning**: Automated vulnerability scanning
