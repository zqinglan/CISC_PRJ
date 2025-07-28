from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from yield_data_fetcher import YieldDataFetcher
from yield_analyzer import YieldAnalyzer
from yield_predictor import YieldPredictor
from enhanced_data_fetcher import EnhancedDataFetcher
import json
from datetime import datetime, timedelta
import os
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import redis
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for all routes

# Initialize components
data_fetcher = YieldDataFetcher()
analyzer = YieldAnalyzer()
predictor = YieldPredictor()
# Initialize enhanced fetcher with Redis disabled if not available
try:
    enhanced_fetcher = EnhancedDataFetcher(
        fred_api_key=os.getenv("FRED_API_KEY"),
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", 6379)),
    )
except Exception as e:
    print(f"Warning: Enhanced data fetcher initialization failed: {e}")
    # Create a minimal enhanced fetcher without Redis
    enhanced_fetcher = EnhancedDataFetcher(
        fred_api_key=os.getenv("FRED_API_KEY"),
        redis_host=None,  # Disable Redis
        redis_port=None,
    )

# Set up enhanced fetcher with base fetcher
enhanced_fetcher.set_base_fetcher(data_fetcher)

# Cached data to avoid frequent API calls
cached_data = {
    "full_data": None,
    "latest_data": None,
    "cycle_data": {},
    "enhanced_data": None,
    "forecasts": {},
    "validation_results": {},
    "last_update": None,
}

# Set up logging
if not os.path.exists("logs"):
    os.makedirs("logs")

file_handler = RotatingFileHandler(
    "logs/yield_analysis.log", maxBytes=10240, backupCount=10
)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
    )
)
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info("Yield Analysis startup")


def refresh_data(force=False):
    """Refresh cached data if needed"""
    current_time = datetime.now()

    # If last update was less than 6 hours ago and not forced, use cached data
    if cached_data["last_update"] and not force:
        time_diff = (current_time - cached_data["last_update"]).total_seconds() / 3600
        if time_diff < 6:
            return

    # Fetch and cache data
    try:
        print("Refreshing data cache...")
        cached_data["full_data"] = data_fetcher.fetch_historical_data()
        cached_data["latest_data"] = data_fetcher.fetch_latest_data(
            days=365
        )  # Get more data to ensure we have enough observations

        # Fetch data for each cutting cycle
        for cycle_name in data_fetcher.cutting_cycles.keys():
            cached_data["cycle_data"][cycle_name] = data_fetcher.fetch_cycle_data(
                cycle_name
            )

        cached_data["last_update"] = current_time
        print("Data cache successfully refreshed")
    except Exception as e:
        print(f"Error refreshing data cache: {str(e)}")


# Helper function to convert DataFrame to JSON-compatible format
def dataframe_to_json(df):
    if df is None or df.empty:
        return []

    # Convert DataFrame to dict with date as string
    result = []
    for date, row in df.iterrows():
        row_dict = {"date": date.strftime("%Y-%m-%d")}
        for col, value in row.items():
            # Handle NaN values
            if pd.isna(value):
                row_dict[col] = None
            else:
                # Convert to percentage for yield values
                if col in analyzer.tenors or col in analyzer.spreads:
                    row_dict[col] = round(
                        float(value) * 100, 2
                    )  # Convert to percentage with 2 decimal places
                else:
                    row_dict[col] = value
        result.append(row_dict)

    return result


# Main routes
@app.route("/")
def index():
    """Main dashboard page"""
    return render_template("index.html")


@app.route("/cycles")
def cycles_page():
    """Cutting cycles analysis page"""
    return render_template("cycles.html")


@app.route("/analysis")
def analysis_page():
    """Custom analysis page"""
    return render_template("analysis.html")


@app.route("/predictive")
def predictive_dashboard():
    """Enhanced predictive dashboard page"""
    return render_template("predictive_dashboard.html")


# API routes
@app.route("/api/health", methods=["GET"])
def health_check():
    """API health check endpoint"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/data/yields", methods=["GET"])
def get_yield_data():
    """Get historical yield data"""
    refresh_data()

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if cached_data["full_data"] is None:
        return jsonify({"error": "Data not available. Please try again later."}), 500

    data = cached_data["full_data"].copy()

    # Filter by date if provided
    if start_date:
        try:
            start_date = pd.to_datetime(start_date)
            data = data[data.index >= start_date]
        except:
            return jsonify({"error": "Invalid start_date format"}), 400

    if end_date:
        try:
            end_date = pd.to_datetime(end_date)
            data = data[data.index <= end_date]
        except:
            return jsonify({"error": "Invalid end_date format"}), 400

    # Convert to JSON format
    json_data = dataframe_to_json(data)

    return jsonify({"data": json_data, "count": len(json_data)})


@app.route("/api/data/latest", methods=["GET"])
def get_latest_data():
    """Get latest yield data"""
    refresh_data()

    days = request.args.get("days", default=90, type=int)

    if cached_data["latest_data"] is None:
        return jsonify({"error": "Data not available. Please try again later."}), 500

    data = cached_data["latest_data"].copy()

    # Convert to JSON format
    json_data = dataframe_to_json(data)

    return jsonify(
        {
            "data": json_data,
            "count": len(json_data),
            "tenors": analyzer.tenors,
            "spreads": analyzer.spreads,
        }
    )


@app.route("/api/cycles", methods=["GET"])
def get_cutting_cycles():
    """Get list of cutting cycles"""
    cycles = []

    for name, details in data_fetcher.cutting_cycles.items():
        cycles.append(
            {
                "id": name,
                "name": details["description"],
                "start_date": details["start"],
                "end_date": details["end"],
            }
        )

    return jsonify({"cycles": cycles})


@app.route("/api/analysis/cycle/<cycle_name>", methods=["GET"])
def analyze_cycle(cycle_name):
    """Analyze a specific cutting cycle"""
    refresh_data()

    if cycle_name not in data_fetcher.cutting_cycles:
        return jsonify({"error": f"Unknown cycle: {cycle_name}"}), 400

    if (
        cycle_name not in cached_data["cycle_data"]
        or cached_data["cycle_data"][cycle_name] is None
    ):
        return (
            jsonify({"error": "Cycle data not available. Please try again later."}),
            500,
        )

    cycle_data = cached_data["cycle_data"][cycle_name].copy()
    cycle_info = data_fetcher.cutting_cycles[cycle_name]

    # Perform analysis
    try:
        analysis = analyzer.analyze_cycle(
            cycle_data, cycle_start=cycle_info["start"], cycle_end=cycle_info["end"]
        )

        # Generate report
        report = analyzer.generate_report(cycle_data, cycle_info["description"])

        # Add cycle metadata
        analysis["cycle_info"] = {
            "name": cycle_info["description"],
            "start_date": cycle_info["start"],
            "end_date": cycle_info["end"],
            "report": report,
        }

        # Add data for charts (convert to percentage)
        analysis["data"] = dataframe_to_json(cycle_data)

        # Add PCA decomposition
        pca_results = analyzer.decompose_yields(cycle_data)
        if pca_results and "components" in pca_results:
            analysis["pca"] = {
                "explained_variance": [
                    round(v * 100, 2) for v in pca_results["explained_variance"]
                ],  # As percentages
                "components": dataframe_to_json(pca_results["components"]),
                "loadings": pca_results["loadings"].to_dict(),
            }

        # Detect regime changes
        regime_changes = analyzer.detect_regime_changes(cycle_data)
        formatted_regimes = []
        for regime in regime_changes:
            formatted_regimes.append(
                {
                    "date": regime["date"].strftime("%Y-%m-%d"),
                    "direction": regime["direction"],
                    "magnitude": round(
                        float(regime["magnitude"]) * 100, 2
                    ),  # As percentage
                    "tenor": regime["tenor"],
                }
            )

        analysis["regimes"] = formatted_regimes

        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500


@app.route("/api/analysis/custom", methods=["POST"])
def custom_analysis():
    """Perform custom analysis on specified date range"""
    refresh_data()

    try:
        # Get parameters from request
        params = request.json
        start_date = params.get("start_date")
        end_date = params.get("end_date")

        if not start_date or not end_date:
            return jsonify({"error": "Start date and end date are required"}), 400

        # Get data
        data = cached_data["full_data"].copy()

        # Filter by date
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
        except Exception as e:
            return jsonify({"error": f"Date filtering error: {str(e)}"}), 400

        if filtered_data.empty:
            return (
                jsonify({"error": "No data available for the specified date range"}),
                404,
            )

        # Perform analysis
        analysis = analyzer.analyze_cycle(filtered_data)

        # Generate report
        report = analyzer.generate_report(filtered_data, "Custom Analysis")

        # Add metadata
        analysis["info"] = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "report": report,
        }

        # Add data for charts
        analysis["data"] = dataframe_to_json(filtered_data)

        # Add PCA decomposition
        pca_results = analyzer.decompose_yields(filtered_data)
        if pca_results and "components" in pca_results:
            analysis["pca"] = {
                "explained_variance": [
                    round(v * 100, 2) for v in pca_results["explained_variance"]
                ],  # As percentages
                "components": dataframe_to_json(pca_results["components"]),
                "loadings": pca_results["loadings"].to_dict(),
            }

        # Detect regime changes
        regime_changes = analyzer.detect_regime_changes(filtered_data)
        formatted_regimes = []
        for regime in regime_changes:
            formatted_regimes.append(
                {
                    "date": regime["date"].strftime("%Y-%m-%d"),
                    "direction": regime["direction"],
                    "magnitude": round(
                        float(regime["magnitude"]) * 100, 2
                    ),  # As percentage
                    "tenor": regime["tenor"],
                }
            )

        analysis["regimes"] = formatted_regimes

        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500


@app.route("/api/charts/yields", methods=["GET"])
def get_yields_chart():
    """Generate yields chart for specified date range or cycle"""
    refresh_data()

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    cycle = request.args.get("cycle")

    # If cycle is specified, use cycle data
    if cycle and cycle in cached_data["cycle_data"]:
        data = cached_data["cycle_data"][cycle].copy()
    elif cached_data["full_data"] is not None:
        data = cached_data["full_data"].copy()

        # Filter by date if provided
        if start_date:
            try:
                start_date = pd.to_datetime(start_date)
                data = data[data.index >= start_date]
            except:
                return jsonify({"error": "Invalid start_date format"}), 400

        if end_date:
            try:
                end_date = pd.to_datetime(end_date)
                data = data[data.index <= end_date]
            except:
                return jsonify({"error": "Invalid end_date format"}), 400
    else:
        return jsonify({"error": "Data not available. Please try again later."}), 500

    # Generate chart
    try:
        title = (
            f"Treasury Yields - {cycle.replace('_', ' ').title()}"
            if cycle
            else "Treasury Yields"
        )
        fig = analyzer.plot_yields(data, title)
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"chart": chart_json})
    except Exception as e:
        return jsonify({"error": f"Chart generation error: {str(e)}"}), 500


@app.route("/api/charts/pca", methods=["GET"])
def get_pca_chart():
    """Generate PCA chart for specified date range"""
    refresh_data()

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if cached_data["full_data"] is None:
        return jsonify({"error": "Data not available. Please try again later."}), 500

    data = cached_data["full_data"].copy()

    # Filter by date if provided
    if start_date:
        try:
            start_date = pd.to_datetime(start_date)
            data = data[data.index >= start_date]
        except:
            return jsonify({"error": "Invalid start_date format"}), 400

    if end_date:
        try:
            end_date = pd.to_datetime(end_date)
            data = data[data.index <= end_date]
        except:
            return jsonify({"error": "Invalid end_date format"}), 400

    # Generate chart
    try:
        fig = analyzer.plot_pca_components(data)
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"chart": chart_json})
    except Exception as e:
        return jsonify({"error": f"Chart generation error: {str(e)}"}), 500


@app.route("/api/charts/regimes", methods=["GET"])
def get_regimes_chart():
    """Generate regime changes chart for specified date range"""
    refresh_data()

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    tenor = request.args.get("tenor", default="10Y")

    if cached_data["full_data"] is None:
        return jsonify({"error": "Data not available. Please try again later."}), 500

    data = cached_data["full_data"].copy()

    # Filter by date if provided
    if start_date:
        try:
            start_date = pd.to_datetime(start_date)
            data = data[data.index >= start_date]
        except:
            return jsonify({"error": "Invalid start_date format"}), 400

    if end_date:
        try:
            end_date = pd.to_datetime(end_date)
            data = data[data.index <= end_date]
        except:
            return jsonify({"error": "Invalid end_date format"}), 400

    # Detect regime changes
    regime_changes = analyzer.detect_regime_changes(data, tenor=tenor)

    # Generate chart
    try:
        fig = analyzer.plot_regime_changes(data, regime_changes)
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"chart": chart_json})
    except Exception as e:
        return jsonify({"error": f"Chart generation error: {str(e)}"}), 500


@app.route("/api/refresh", methods=["POST"])
def force_refresh():
    """Force refresh of cached data"""
    try:
        refresh_data(force=True)
        return jsonify(
            {
                "status": "success",
                "message": "Data cache refreshed",
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        return (
            jsonify({"status": "error", "message": f"Error refreshing data: {str(e)}"}),
            500,
        )


# New predictive modeling endpoints
@app.route("/api/forecast/generate", methods=["POST"])
def generate_forecast():
    """Generate yield forecasts using VAR models"""
    try:
        params = request.json or {}
        forecast_horizon = params.get("horizon", 5)
        confidence_level = params.get("confidence_level", 0.95)
        tenors = params.get("tenors", ["3M", "2Y", "5Y", "10Y", "30Y"])

        # Get latest data
        refresh_data()
        if cached_data["latest_data"] is None:
            return jsonify({"error": "No data available for forecasting"}), 500

        # Check if we have enough data for forecasting
        data = cached_data["latest_data"]
        if len(data) < 7:  # Reduced from 100 to 7 observations
            return (
                jsonify(
                    {
                        "error": f"Insufficient data for forecasting. Need at least 7 observations, got {len(data)}."
                    }
                ),
                500,
            )

        # Check for missing values
        available_tenors = [t for t in tenors if t in data.columns]
        if len(available_tenors) < 2:
            return jsonify({"error": "Insufficient yield data for VAR modeling"}), 500

        # Generate forecast with available tenors
        forecast_results = predictor.generate_forecast(
            data,
            tenors=available_tenors,
            forecast_horizon=forecast_horizon,
            confidence_level=confidence_level,
        )

        if forecast_results:
            # Cache the forecast
            forecast_key = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M')}"
            cached_data["forecasts"][forecast_key] = forecast_results

            # Get forecast summary
            summary = predictor.get_forecast_summary(forecast_results)

            return jsonify(
                {
                    "forecast_id": forecast_key,
                    "forecast": forecast_results,
                    "summary": summary,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            return (
                jsonify(
                    {
                        "error": "Failed to generate forecast. Please try with different parameters."
                    }
                ),
                500,
            )

    except Exception as e:
        app.logger.error(f"Forecast generation error: {str(e)}")
        return jsonify({"error": f"Forecast error: {str(e)}"}), 500


@app.route("/api/forecast/validate", methods=["POST"])
def validate_forecast():
    """Perform walk-forward validation of forecasting models"""
    try:
        params = request.json or {}
        # Adjust train window based on available data
        data_length = len(cached_data["full_data"])
        default_train_window = min(
            252, max(50, data_length // 2)
        )  # Adaptive train window
        default_test_window = min(21, max(5, data_length // 10))  # Adaptive test window

        train_window = params.get("train_window", default_train_window)
        test_window = params.get("test_window", default_test_window)
        forecast_horizon = params.get("forecast_horizon", 5)
        tenors = params.get("tenors", ["3M", "2Y", "5Y", "10Y", "30Y"])

        # Get historical data
        refresh_data()
        if cached_data["full_data"] is None:
            return jsonify({"error": "No historical data available"}), 500

        # Perform validation
        validation_results = predictor.rolling_walk_forward_validation(
            cached_data["full_data"],
            tenors=tenors,
            train_window=train_window,
            test_window=test_window,
            forecast_horizon=forecast_horizon,
        )

        # Cache validation results
        validation_key = f"validation_{datetime.now().strftime('%Y%m%d_%H%M')}"
        cached_data["validation_results"][validation_key] = validation_results

        return jsonify(
            {
                "validation_id": validation_key,
                "results": validation_results,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        app.logger.error(f"Validation error: {str(e)}")
        return jsonify({"error": f"Validation error: {str(e)}"}), 500


@app.route("/api/forecast/<forecast_id>", methods=["GET"])
def get_forecast(forecast_id):
    """Get a specific forecast by ID"""
    if forecast_id not in cached_data["forecasts"]:
        return jsonify({"error": "Forecast not found"}), 404

    forecast_results = cached_data["forecasts"][forecast_id]
    summary = predictor.get_forecast_summary(forecast_results)

    return jsonify(
        {"forecast_id": forecast_id, "forecast": forecast_results, "summary": summary}
    )


@app.route("/api/forecast/<forecast_id>/chart", methods=["GET"])
def get_forecast_chart(forecast_id):
    """Get forecast visualization chart"""
    try:
        if forecast_id not in cached_data["forecasts"]:
            return jsonify({"error": "Forecast not found"}), 404

        forecast_results = cached_data["forecasts"][forecast_id]

        # Get historical data for comparison
        refresh_data()
        if cached_data["latest_data"] is None:
            return jsonify({"error": "No historical data available"}), 500

        # Generate chart
        fig = predictor.plot_forecast(cached_data["latest_data"], forecast_results)

        if fig:
            chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return jsonify({"chart": chart_json})
        else:
            return jsonify({"error": "Failed to generate chart"}), 500
    except Exception as e:
        app.logger.error(f"Forecast chart error: {str(e)}")
        return jsonify({"error": f"Chart generation error: {str(e)}"}), 500


@app.route("/api/validation/<validation_id>/chart", methods=["GET"])
def get_validation_chart(validation_id):
    """Get validation results visualization chart"""
    try:
        if validation_id not in cached_data["validation_results"]:
            return jsonify({"error": "Validation results not found"}), 404

        validation_results = cached_data["validation_results"][validation_id]

        # Generate chart
        fig = predictor.plot_validation_results(validation_results)

        if fig:
            chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return jsonify({"chart": chart_json})
        else:
            return jsonify({"error": "Failed to generate chart"}), 500
    except Exception as e:
        app.logger.error(f"Validation chart error: {str(e)}")
        return jsonify({"error": f"Chart generation error: {str(e)}"}), 500


# Enhanced data endpoints
@app.route("/api/data/enhanced", methods=["GET"])
def get_enhanced_data():
    """Get enhanced data from multiple sources"""
    try:
        days = request.args.get("days", default=90, type=int)

        # Check cache first (only if Redis is available)
        cache_key = f"enhanced_data_{days}"
        if enhanced_fetcher.redis_available and enhanced_fetcher.redis_client:
            try:
                cached_result = enhanced_fetcher.redis_client.get(cache_key)
                if cached_result:
                    return jsonify(json.loads(cached_result))
            except Exception as e:
                app.logger.warning(f"Redis cache access failed: {str(e)}")

        # Fetch enhanced data
        try:
            enhanced_data = enhanced_fetcher.get_latest_data_sync(days)
        except Exception as e:
            app.logger.warning(f"Enhanced data fetch failed: {str(e)}")
            # Fall back to basic data
            refresh_data()
            if cached_data["latest_data"] is not None:
                enhanced_data = {"treasury": cached_data["latest_data"]}
            else:
                return jsonify({"error": "No data available"}), 500

        # Merge data sources
        merged_data = enhanced_fetcher.merge_data_sources(enhanced_data)

        if merged_data is not None and not merged_data.empty:
            # Convert to JSON format
            json_data = dataframe_to_json(merged_data)

            result = {
                "data": json_data,
                "count": len(json_data),
                "sources": enhanced_fetcher.get_data_summary(enhanced_data),
                "cache_status": enhanced_fetcher.get_cache_status(),
            }

            # Cache the result (only if Redis is available)
            if enhanced_fetcher.redis_available and enhanced_fetcher.redis_client:
                try:
                    enhanced_fetcher.redis_client.setex(
                        cache_key,
                        enhanced_fetcher.cache_ttl["treasury"],
                        json.dumps(result),
                    )
                except Exception as e:
                    app.logger.warning(f"Redis cache write failed: {str(e)}")

            return jsonify(result)
        else:
            return jsonify({"error": "No enhanced data available"}), 500

    except Exception as e:
        app.logger.error(f"Enhanced data error: {str(e)}")
        return jsonify({"error": f"Enhanced data error: {str(e)}"}), 500


@app.route("/api/cache/status", methods=["GET"])
def get_cache_status():
    """Get cache status and statistics"""
    try:
        return jsonify(enhanced_fetcher.get_cache_status())
    except Exception as e:
        app.logger.error(f"Cache status error: {str(e)}")
        return jsonify({"status": "unavailable", "total_keys": 0, "error": str(e)})


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """Clear cache entries"""
    try:
        params = request.json or {}
        cache_type = params.get("type")  # Optional: specific cache type

        success = enhanced_fetcher.clear_cache(cache_type)

        return jsonify(
            {
                "status": "success" if success else "no_cache",
                "message": (
                    f"Cache cleared for type: {cache_type}"
                    if cache_type
                    else "All cache cleared"
                ),
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        return jsonify({"error": f"Cache clear error: {str(e)}"}), 500


# Alert system endpoints
@app.route("/api/alerts/check", methods=["POST"])
def check_alerts():
    """Check for yield curve deviations and generate alerts"""
    try:
        params = request.json or {}
        threshold = params.get("threshold", 0.02)  # 2% deviation threshold

        # Get latest data
        refresh_data()
        if cached_data["latest_data"] is None:
            return jsonify({"error": "No data available for alert checking"}), 500

        # Get latest forecast
        latest_forecast_id = None
        if cached_data["forecasts"]:
            latest_forecast_id = max(cached_data["forecasts"].keys())

        alerts = []

        if latest_forecast_id:
            forecast_results = cached_data["forecasts"][latest_forecast_id]
            forecast = forecast_results["forecast"]

            # Compare actual vs forecasted values
            latest_actual = cached_data["latest_data"].iloc[-1]

            for i, tenor in enumerate(forecast_results["model_info"]["tenors"]):
                if tenor in latest_actual.index and i < len(forecast[0]):
                    actual_value = latest_actual[tenor]
                    forecast_value = forecast[0][i]  # First forecast period

                    deviation = abs(actual_value - forecast_value) / actual_value

                    if deviation > threshold:
                        alerts.append(
                            {
                                "tenor": tenor,
                                "actual": actual_value * 100,  # Convert to percentage
                                "forecast": forecast_value * 100,
                                "deviation": deviation * 100,
                                "severity": (
                                    "high" if deviation > threshold * 2 else "medium"
                                ),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

        return jsonify(
            {
                "alerts": alerts,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        app.logger.error(f"Alert check error: {str(e)}")
        return jsonify({"error": f"Alert check error: {str(e)}"}), 500


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    # Initial data load
    refresh_data(force=True)

    # Run the app
    app.run(debug=True, host="0.0.0.0", port=5000)
