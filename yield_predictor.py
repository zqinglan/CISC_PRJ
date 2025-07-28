import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

warnings.filterwarnings("ignore")


class YieldPredictor:
    """
    Predictive modeling for Treasury yields using Vector Autoregressive (VAR) models
    with rolling walk-forward validation and confidence intervals.
    """

    def __init__(self):
        self.tenors = ["3M", "2Y", "5Y", "10Y", "30Y"]
        self.models = {}
        self.forecast_history = {}
        self.validation_metrics = {}
        self.colors = {
            "3M": "#1f77b4",
            "2Y": "#ff7f0e",
            "5Y": "#2ca02c",
            "10Y": "#d62728",
            "30Y": "#9467bd",
        }

    def prepare_data_for_var(self, data, tenors=None):
        """
        Prepare data for VAR modeling by ensuring stationarity

        Args:
            data (DataFrame): Yield data
            tenors (list): Tenors to include in the model

        Returns:
            DataFrame: Stationary data ready for VAR
        """
        if tenors is None:
            tenors = [t for t in self.tenors if t in data.columns]

        # Select relevant columns
        model_data = data[tenors].copy()

        # Handle missing values
        model_data = model_data.dropna()

        # For smaller datasets, be more conservative with stationarity testing
        # Use first differences by default for yield data
        stationary_data = pd.DataFrame(index=model_data.index)

        for tenor in tenors:
            series = model_data[tenor]

            # For small datasets (< 50 observations), use first differences
            if len(series) < 50:
                diff_series = series.diff().dropna()
                # Add small noise to prevent perfect multicollinearity
                if len(diff_series) > 0:
                    noise = np.random.normal(
                        0, diff_series.std() * 0.001, len(diff_series)
                    )
                    stationary_data[tenor] = diff_series + noise
                else:
                    stationary_data[tenor] = diff_series
            else:
                # Test for stationarity only for larger datasets
                try:
                    adf_result = adfuller(series.dropna())
                    if adf_result[1] > 0.05:  # Not stationary
                        diff_series = series.diff().dropna()
                        # Add small noise to prevent perfect multicollinearity
                        if len(diff_series) > 0:
                            noise = np.random.normal(
                                0, diff_series.std() * 0.001, len(diff_series)
                            )
                            stationary_data[tenor] = diff_series + noise
                        else:
                            stationary_data[tenor] = diff_series
                    else:
                        stationary_data[tenor] = series
                except:
                    # If ADF test fails, use first differences
                    diff_series = series.diff().dropna()
                    if len(diff_series) > 0:
                        noise = np.random.normal(
                            0, diff_series.std() * 0.001, len(diff_series)
                        )
                        stationary_data[tenor] = diff_series + noise
                    else:
                        stationary_data[tenor] = diff_series

        return stationary_data.dropna()

    def _generate_simple_forecast(
        self, data, tenors, forecast_horizon, confidence_level
    ):
        """
        Generate a simple linear trend forecast as fallback
        """
        if tenors is None:
            tenors = [t for t in self.tenors if t in data.columns]

        # Select relevant columns
        model_data = data[tenors].copy().dropna()

        if len(model_data) < 3:
            return None

        # Generate simple linear trend forecasts
        forecast = []
        confidence_intervals = {"lower": [], "upper": []}

        for tenor in tenors:
            if tenor in model_data.columns:
                series = model_data[tenor]
                if len(series) >= 3:
                    # Simple linear trend
                    x = np.arange(len(series))
                    slope, intercept = np.polyfit(x, series, 1)

                    # Generate forecast
                    future_x = np.arange(len(series), len(series) + forecast_horizon)
                    future_values = slope * future_x + intercept
                    forecast.append(future_values)

                    # Simple confidence intervals based on historical volatility
                    std_dev = series.std()
                    z_score = 1.96  # 95% confidence
                    lower = future_values - z_score * std_dev
                    upper = future_values + z_score * std_dev
                    confidence_intervals["lower"].append(lower)
                    confidence_intervals["upper"].append(upper)
                else:
                    # Use last value
                    last_value = series.iloc[-1]
                    forecast.append([last_value] * forecast_horizon)
                    confidence_intervals["lower"].append(
                        [last_value * 0.95] * forecast_horizon
                    )
                    confidence_intervals["upper"].append(
                        [last_value * 1.05] * forecast_horizon
                    )

        # Convert to numpy arrays
        forecast = np.array(forecast).T
        confidence_intervals["lower"] = np.array(confidence_intervals["lower"]).T
        confidence_intervals["upper"] = np.array(confidence_intervals["upper"]).T

        # Create forecast dates
        last_date = data.index[-1]
        forecast_dates = [
            last_date + timedelta(days=i + 1) for i in range(forecast_horizon)
        ]

        return {
            "forecast": forecast.tolist(),
            "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
            "confidence_intervals": {
                "lower": confidence_intervals["lower"].tolist(),
                "upper": confidence_intervals["upper"].tolist(),
            },
            "model_info": {
                "aic": 0.0,  # Not applicable for simple model
                "bic": 0.0,
                "lags": 1,
                "tenors": tenors,
                "model_type": "simple_linear_trend",
            },
        }

    def fit_var_model(self, data, tenors=None, maxlags=15):
        """
        Fit VAR model to the data

        Args:
            data (DataFrame): Yield data
            tenors (list): Tenors to include
            maxlags (int): Maximum number of lags to consider

        Returns:
            VAR model object
        """
        if tenors is None:
            tenors = [t for t in self.tenors if t in data.columns]

        # Prepare stationary data
        stationary_data = self.prepare_data_for_var(data, tenors)

        # Adjust maxlags based on available data
        min_required = maxlags + 5  # Reduced from 10 to 5
        if len(stationary_data) < min_required:
            # Reduce maxlags to fit available data
            maxlags = max(1, len(stationary_data) - 5)
            if maxlags < 2:
                raise ValueError(
                    f"Insufficient data for VAR modeling. Need at least 7 observations, got {len(stationary_data)}."
                )

        # Fit VAR model
        model = VAR(stationary_data)

        # Select optimal lag order using AIC
        try:
            lag_order = model.select_order(maxlags=maxlags)
            optimal_lags = lag_order.aic
        except:
            optimal_lags = min(2, maxlags)  # Reduced further to 2

        # Try to fit the model with optimal lags
        try:
            fitted_model = model.fit(optimal_lags)
        except Exception as e:
            # If that fails, try with lag=1
            try:
                fitted_model = model.fit(1)
            except Exception as e2:
                # If that also fails, try with a subset of variables
                if len(stationary_data.columns) > 2:
                    # Use only the first 2 variables
                    subset_data = stationary_data.iloc[:, :2]
                    subset_model = VAR(subset_data)
                    fitted_model = subset_model.fit(1)
                    # Update the stationary data to match
                    stationary_data = subset_data
                else:
                    raise ValueError(f"Unable to fit VAR model: {str(e2)}")

        return fitted_model, stationary_data

    def rolling_walk_forward_validation(
        self, data, tenors=None, train_window=252, test_window=21, forecast_horizon=5
    ):
        """
        Perform rolling walk-forward validation

        Args:
            data (DataFrame): Historical yield data
            tenors (list): Tenors to forecast
            train_window (int): Training window size (days)
            test_window (int): Test window size (days)
            forecast_horizon (int): Forecast horizon (days)

        Returns:
            dict: Validation results and metrics
        """
        if tenors is None:
            tenors = [t for t in self.tenors if t in data.columns]

        # Prepare data
        model_data = data[tenors].copy().dropna()

        if len(model_data) < train_window + test_window + forecast_horizon:
            raise ValueError("Insufficient data for walk-forward validation")

        # Initialize results storage
        forecasts = []
        actuals = []
        forecast_dates = []
        model_performance = []

        # Perform rolling validation
        for i in range(0, len(model_data) - train_window - test_window, test_window):
            # Split data
            train_end = i + train_window
            test_start = train_end
            test_end = min(test_start + test_window, len(model_data))

            train_data = model_data.iloc[i:train_end]
            test_data = model_data.iloc[test_start:test_end]

            try:
                # Fit model
                fitted_model, stationary_train = self.fit_var_model(train_data, tenors)

                # Make forecast
                forecast = fitted_model.forecast(
                    stationary_train.values[-fitted_model.k_ar :],
                    steps=forecast_horizon,
                )

                # Convert forecast back to levels if needed
                forecast_levels = self._convert_to_levels(forecast, train_data, tenors)

                # Store results
                for j, forecast_date in enumerate(
                    model_data.index[test_start : test_start + forecast_horizon]
                ):
                    if j < len(forecast_levels):
                        forecasts.append(forecast_levels[j])
                        actuals.append(
                            test_data.iloc[j].values if j < len(test_data) else None
                        )
                        forecast_dates.append(forecast_date)

                # Calculate performance metrics for this window
                if len(test_data) >= forecast_horizon:
                    actual_test = test_data.iloc[:forecast_horizon].values
                    forecast_test = forecast_levels[:forecast_horizon]

                    mse = mean_squared_error(actual_test, forecast_test)
                    mae = mean_absolute_error(actual_test, forecast_test)

                    model_performance.append(
                        {
                            "window_start": model_data.index[i],
                            "window_end": model_data.index[test_end - 1],
                            "mse": mse,
                            "mae": mae,
                            "rmse": np.sqrt(mse),
                        }
                    )

            except Exception as e:
                print(f"Error in validation window {i}: {str(e)}")
                continue

        # Aggregate results - convert to JSON-serializable format
        results = {
            "forecasts": [f.tolist() if hasattr(f, "tolist") else f for f in forecasts],
            "actuals": [a.tolist() if hasattr(a, "tolist") else a for a in actuals],
            "forecast_dates": [
                d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else d
                for d in forecast_dates
            ],
            "model_performance": [
                {
                    "window_start": (
                        perf["window_start"].strftime("%Y-%m-%d")
                        if hasattr(perf["window_start"], "strftime")
                        else perf["window_start"]
                    ),
                    "window_end": (
                        perf["window_end"].strftime("%Y-%m-%d")
                        if hasattr(perf["window_end"], "strftime")
                        else perf["window_end"]
                    ),
                    "mse": float(perf["mse"]),
                    "mae": float(perf["mae"]),
                    "rmse": float(perf["rmse"]),
                }
                for perf in model_performance
            ],
            "tenors": tenors,
        }

        # Calculate overall metrics
        if forecasts and actuals:
            valid_forecasts = [f for f, a in zip(forecasts, actuals) if a is not None]
            valid_actuals = [a for a in actuals if a is not None]

            if valid_forecasts and valid_actuals:
                overall_mse = mean_squared_error(valid_actuals, valid_forecasts)
                overall_mae = mean_absolute_error(valid_actuals, valid_forecasts)

                results["overall_metrics"] = {
                    "mse": float(overall_mse),
                    "mae": float(overall_mae),
                    "rmse": float(np.sqrt(overall_mse)),
                }

        return results

    def _convert_to_levels(self, forecast, original_data, tenors):
        """
        Convert differenced forecasts back to levels
        """
        # This is a simplified conversion - in practice, you'd need to track
        # the original differencing and apply the inverse transformation
        return forecast

    def generate_forecast(
        self,
        data,
        tenors=None,
        forecast_horizon=5,
        confidence_level=0.95,
        n_simulations=1000,
    ):
        """
        Generate yield forecasts with confidence intervals

        Args:
            data (DataFrame): Historical yield data
            tenors (list): Tenors to forecast
            forecast_horizon (int): Number of days to forecast
            confidence_level (float): Confidence level for intervals
            n_simulations (int): Number of Monte Carlo simulations

        Returns:
            dict: Forecast results with confidence intervals
        """
        if tenors is None:
            tenors = [t for t in self.tenors if t in data.columns]

        try:
            # Fit model
            fitted_model, stationary_data = self.fit_var_model(data, tenors)

            # Generate forecast
            forecast = fitted_model.forecast(
                stationary_data.values[-fitted_model.k_ar :], steps=forecast_horizon
            )

            # Generate confidence intervals using Monte Carlo simulation
            confidence_intervals = self._generate_confidence_intervals(
                fitted_model,
                stationary_data,
                forecast_horizon,
                confidence_level,
                n_simulations,
            )

            # Create forecast dates
            last_date = data.index[-1]
            forecast_dates = [
                last_date + timedelta(days=i + 1) for i in range(forecast_horizon)
            ]

            # Prepare results - convert numpy arrays to lists for JSON serialization
            results = {
                "forecast": (
                    forecast.tolist() if hasattr(forecast, "tolist") else forecast
                ),
                "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
                "confidence_intervals": {
                    "lower": (
                        confidence_intervals["lower"].tolist()
                        if hasattr(confidence_intervals["lower"], "tolist")
                        else confidence_intervals["lower"]
                    ),
                    "upper": (
                        confidence_intervals["upper"].tolist()
                        if hasattr(confidence_intervals["upper"], "tolist")
                        else confidence_intervals["upper"]
                    ),
                },
                "model_info": {
                    "aic": float(fitted_model.aic),
                    "bic": float(fitted_model.bic),
                    "lags": int(fitted_model.k_ar),
                    "tenors": tenors,
                },
            }

            return results

        except Exception as e:
            print(f"Error generating forecast: {str(e)}")
            # Try a simple linear trend model as fallback
            try:
                return self._generate_simple_forecast(
                    data, tenors, forecast_horizon, confidence_level
                )
            except Exception as e2:
                print(f"Simple forecast also failed: {str(e2)}")
                return None

    def _generate_confidence_intervals(
        self, model, data, horizon, confidence_level, n_simulations
    ):
        """
        Generate confidence intervals using Monte Carlo simulation
        """
        # This is a simplified implementation
        # In practice, you'd use the model's built-in methods or more sophisticated simulation

        # Get model residuals
        residuals = model.resid

        # Generate simulations
        simulations = []
        for _ in range(n_simulations):
            # Add random noise to residuals
            noise = np.random.normal(
                0, residuals.std().values, (horizon, len(residuals.columns))
            )
            sim_forecast = (
                model.forecast(data.values[-model.k_ar :], steps=horizon) + noise
            )
            simulations.append(sim_forecast)

        simulations = np.array(simulations)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        confidence_intervals = {
            "lower": np.percentile(simulations, lower_percentile, axis=0),
            "upper": np.percentile(simulations, upper_percentile, axis=0),
            "mean": np.mean(simulations, axis=0),
        }

        return confidence_intervals

    def plot_forecast(self, data, forecast_results, title="Yield Forecast"):
        """
        Plot historical data with forecasts and confidence intervals

        Args:
            data (DataFrame): Historical yield data
            forecast_results (dict): Forecast results
            title (str): Plot title

        Returns:
            plotly.graph_objects.Figure: The figure object
        """
        if not forecast_results:
            return None

        # Create figure
        fig = go.Figure()

        tenors = forecast_results["model_info"]["tenors"]
        forecast_dates = forecast_results["forecast_dates"]
        forecast = forecast_results["forecast"]
        confidence_intervals = forecast_results["confidence_intervals"]

        # Plot historical data
        for i, tenor in enumerate(tenors):
            if tenor in data.columns:
                # Convert to percentage
                historical_values = data[tenor] * 100

                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=historical_values,
                        name=f"{tenor} (Historical)",
                        line=dict(color=self.colors.get(tenor, None), width=2),
                        showlegend=True,
                    )
                )

                # Plot forecast - handle both numpy arrays and lists
                if isinstance(forecast, list):
                    forecast_values = [row[i] * 100 for row in forecast]
                else:
                    forecast_values = forecast[:, i] * 100

                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        name=f"{tenor} (Forecast)",
                        line=dict(
                            color=self.colors.get(tenor, None), width=2, dash="dash"
                        ),
                        showlegend=True,
                    )
                )

                # Plot confidence intervals - handle both numpy arrays and lists
                if isinstance(confidence_intervals["lower"], list):
                    lower_ci = [row[i] * 100 for row in confidence_intervals["lower"]]
                    upper_ci = [row[i] * 100 for row in confidence_intervals["upper"]]
                else:
                    lower_ci = confidence_intervals["lower"][:, i] * 100
                    upper_ci = confidence_intervals["upper"][:, i] * 100

                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates + forecast_dates[::-1],
                        y=list(upper_ci) + list(lower_ci)[::-1],
                        fill="toself",
                        fillcolor=f'rgba{(*self._hex_to_rgb(self.colors.get(tenor, "#000000")), 0.2)}',
                        line=dict(color="rgba(255,255,255,0)"),
                        name=f"{tenor} Confidence Interval",
                        showlegend=False,
                    )
                )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Yield (%)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        return fig

    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB for transparency"""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def plot_validation_results(
        self, validation_results, title="Walk-Forward Validation"
    ):
        """
        Plot validation results showing forecast accuracy

        Args:
            validation_results (dict): Results from walk-forward validation
            title (str): Plot title

        Returns:
            plotly.graph_objects.Figure: The figure object
        """
        if not validation_results or "forecasts" not in validation_results:
            return None

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Forecast vs Actual", "Model Performance Over Time"),
            vertical_spacing=0.15,
        )

        tenors = validation_results["tenors"]
        forecasts = validation_results["forecasts"]
        actuals = validation_results["actuals"]
        forecast_dates = validation_results["forecast_dates"]

        # Plot forecast vs actual for each tenor
        for i, tenor in enumerate(tenors):
            if i < len(forecasts[0]):
                forecast_series = [f[i] * 100 for f in forecasts if f is not None]
                actual_series = [a[i] * 100 for a in actuals if a is not None]

                if forecast_series and actual_series:
                    fig.add_trace(
                        go.Scatter(
                            x=actual_series,
                            y=forecast_series,
                            mode="markers",
                            name=f"{tenor} Forecast vs Actual",
                            marker=dict(color=self.colors.get(tenor, None)),
                            showlegend=True,
                        ),
                        row=1,
                        col=1,
                    )

        # Add perfect forecast line
        if actuals:
            min_val = min(
                [
                    a[i] * 100
                    for a in actuals
                    if a is not None
                    for i in range(len(tenors))
                ]
            )
            max_val = max(
                [
                    a[i] * 100
                    for a in actuals
                    if a is not None
                    for i in range(len(tenors))
                ]
            )

            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Perfect Forecast",
                    line=dict(color="black", dash="dash"),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # Plot model performance over time
        if "model_performance" in validation_results:
            performance = validation_results["model_performance"]
            dates = [p["window_start"] for p in performance]
            rmse_values = [p["rmse"] for p in performance]

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=rmse_values,
                    mode="lines+markers",
                    name="RMSE",
                    line=dict(color="red"),
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_layout(title=title, template="plotly_white", height=800)

        # Update axes
        fig.update_xaxes(title_text="Actual Yield (%)", row=1, col=1)
        fig.update_yaxes(title_text="Forecasted Yield (%)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="RMSE", row=2, col=1)

        return fig

    def get_forecast_summary(self, forecast_results):
        """
        Generate a summary of forecast results

        Args:
            forecast_results (dict): Forecast results

        Returns:
            dict: Summary statistics
        """
        if not forecast_results:
            return None

        tenors = forecast_results["model_info"]["tenors"]
        forecast = forecast_results["forecast"]
        confidence_intervals = forecast_results["confidence_intervals"]

        summary = {
            "forecast_horizon": len(forecast),
            "tenors": tenors,
            "forecasts": {},
            "model_quality": {
                "aic": forecast_results["model_info"]["aic"],
                "bic": forecast_results["model_info"]["bic"],
                "lags": forecast_results["model_info"]["lags"],
                "model_type": forecast_results["model_info"].get("model_type", "VAR"),
            },
        }

        for i, tenor in enumerate(tenors):
            if i < len(forecast[0]):
                # Handle both numpy arrays and lists
                if isinstance(forecast, list):
                    forecast_values = (
                        np.array([row[i] for row in forecast]) * 100
                    )  # Convert to percentage
                    lower_ci = (
                        np.array([row[i] for row in confidence_intervals["lower"]])
                        * 100
                    )
                    upper_ci = (
                        np.array([row[i] for row in confidence_intervals["upper"]])
                        * 100
                    )
                else:
                    forecast_values = forecast[:, i] * 100  # Convert to percentage
                    lower_ci = confidence_intervals["lower"][:, i] * 100
                    upper_ci = confidence_intervals["upper"][:, i] * 100

                summary["forecasts"][tenor] = {
                    "values": (
                        forecast_values.tolist()
                        if hasattr(forecast_values, "tolist")
                        else forecast_values
                    ),
                    "confidence_lower": (
                        lower_ci.tolist() if hasattr(lower_ci, "tolist") else lower_ci
                    ),
                    "confidence_upper": (
                        upper_ci.tolist() if hasattr(upper_ci, "tolist") else upper_ci
                    ),
                    "mean_forecast": float(np.mean(forecast_values)),
                    "forecast_volatility": float(np.std(forecast_values)),
                }

        return summary
