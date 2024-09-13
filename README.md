# Customer Lifetime Value (CLV) Migration Model Calculation

## Overview

This project provides a Python implementation of a Customer Lifetime Value (CLV) migration model calculation. The model computes CLV using various data frames that inform the calculation through rates, contributions, retentions, and other relevant metrics. It includes methods to handle future projections and outlier remediation, ensuring accurate and reliable results.

## Features

- **Data Frame Splitting**: Separates the main DataFrame into rule, retention, and contribution DataFrames for streamlined processing.
- **Retention Rate Matrix Calculation**: Computes retention rates while handling outliers and ensuring data integrity.
- **Contribution Value Matrix Calculation**: Calculates ratio matrices from contribution data, considering only non-zero values.
- **CLV Year Matrix Calculation**: Computes the Customer Lifetime Value for each period based on retention and contribution matrices.
- **Future Projection**: Extends matrices to predict future retention rates, contributions, and CLV over specified future periods.
- **Outlier Handling**: Implements an outlier remedy method to handle anomalies in the data, ensuring robust calculations.

## Installation

Ensure you have Python 3.x installed along with the necessary packages:

```bash
pip install numpy pandas
```

git clone https://github.com/VustyWu/CLV-Migration-model/tree/main


Prepare your data frames:

- **rule_df**: DataFrame containing rules for calculating weights.
- **retention_df**: DataFrame containing retention rates.
- **rate_df**: DataFrame containing rates.
- **contribution_df**: DataFrame containing contributions.
- **ratio_df**: DataFrame containing ratios.
- **clv_df**: DataFrame containing initial CLV values.


Interpretation

The **CLVCalculator** processes customer data to calculate the Customer Lifetime Value over a specified number of periods, including future projections. Here's how to interpret the outputs:

- **Retention Matrix (retention_df)**: Shows the retention rates at each period for different customer segments.
- **Rate Matrix (rate_df)**: Provides the rate of change in retention between periods, adjusted for outliers.
- **Contribution Matrix (contribution_df)**: Displays the average contribution (e.g., revenue) per customer in each period.
- **Ratio Matrix (ratio_df)**: Indicates the ratio of contributions between periods, highlighting growth or decline trends.
- **CLV Matrix (clv_df)**: Contains the calculated CLV for each customer segment over time.
- **Future Matrices**: Extend the initial matrices to include predicted values for future periods based on specified assumptions.

Understanding the Calculations
- **Weights Calculation*: The model calculates weights for each value in the number DataFrame based on the rule DataFrame, ensuring that retention and rate calculations are accurate.
- **Outlier Remediation*: The outlier handling method adjusts extreme values in the rate and ratio matrices to prevent them from skewing the results.
- **Future Projections*: By providing future retention rates and ratios, the model predicts how the CLV might change over upcoming periods.

  You may follow the demo with specific examples

