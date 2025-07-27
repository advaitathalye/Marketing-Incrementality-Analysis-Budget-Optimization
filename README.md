# Marketing-Incrementality-Analysis-Budget-Optimization

Building the future of marketing measurement through causal inference and statistical rigor
**Executive Summary**
Traditional marketing attribution is fundamentally broken. Last-click models and multi-touch attribution confuse correlation with causation, leading to billions in wasted ad spend annually.

This project implements incrementality measurement, the gold standard used by companies like AG1, Caraway, and Sonos to measure the true causal impact of marketing investments and optimize budget allocation scientifically.

**Key Results:**
🔍 Identified 52-61% attribution bias across all channels in traditional models
💰 Generated $33.6M budget reallocation recommendations (Email: +$8.7M, TV: -$13.5M)
📈 Email severely underfunded: Highest incremental ROI (1.71x) with only 3% of budget
🧪 Geo-lift testing shows significant regional impact: APAC (107% lift), EU (143% lift), US (127% lift)

**The Incrementality Revolution**

**What is Incrementality?**
Incrementality answers the fundamental question: "What would have happened without our ads?"

Instead of assuming all conversions are caused by marketing, incrementality uses controlled experiments to isolate true causal impact.

Incremental Lift = (Treatment Conversions - Control Conversions) / Control Conversions

**Why It Matters**
Traditional Attribution: "Our TV campaign drove 10,000 conversions!"
Incrementality: "Our TV campaign drove 3,200 additional conversions (6,800 would have happened anyway)"
This distinction is worth millions in budget optimization.

**📊 Dataset & Methodology**
Scale: 19,656 marketing touchpoints across 6 months
Channels: Search, Social, Display, Email, Affiliate, TV
Geography: US, EU, APAC regions with treatment/control splits
Experiment Design: Geo-lift testing with statistical controls Multi-Touch Attribution: 1,000 customer journey simulations across 5 channels
Key Variables:
spend, impressions, clicks, conversions
incremental_conversions (derived from holdout tests)
experiment_group (Treatment vs Control)
geo_region, device, channel

**Analysis Framework**
1. Traditional vs Incremental ROI Comparison
```# Aggregate data by channel
channel_summary = df.groupby('channel').agg({
    'spend': 'sum',
    'conversions': 'sum',
    'incremental_conversions': 'sum',
    'revenue': 'sum'
}).reset_index()

# Traditional ROI
channel_summary['traditional_roi'] = channel_summary['revenue'] / channel_summary['spend']

# Incremental ROI
channel_summary['incremental_revenue'] = (
    channel_summary['incremental_conversions'] *
    (channel_summary['revenue'] / channel_summary['conversions'])
)
channel_summary['incremental_roi'] = channel_summary['incremental_revenue'] / channel_summary['spend']

# Visualization: Traditional vs Incremental ROI
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.bar(channel_summary['channel'], channel_summary['traditional_roi'], alpha=0.6, label='Traditional ROI')
plt.bar(channel_summary['channel'], channel_summary['incremental_roi'], alpha=0.6, label='Incremental ROI')
plt.ylabel('ROI')
plt.title('Traditional vs Incremental ROI by Channel')
plt.legend()
plt.show()
```

2. Budget Allocation: Current vs Recommended
```# Calculate recommended spend
total_budget = channel_summary['spend'].sum()
channel_summary['recommended_spend'] = (
    channel_summary['incremental_roi'] /
    channel_summary['incremental_roi'].sum()
) * total_budget

# Visualization: Current vs Recommended Budget Allocation
plt.figure(figsize=(10,6))
plt.bar(channel_summary['channel'], channel_summary['spend'], alpha=0.6, label='Current Spend')
plt.bar(channel_summary['channel'], channel_summary['recommended_spend'], alpha=0.6, label='Recommended Spend')
plt.ylabel('Budget ($)')
plt.title('Budget Allocation: Current vs Recommended')
plt.legend()
plt.show()
```

3.Incrementality Lift by Region
```# Group by region and experiment group
region_group = df.groupby(['geo_region', 'experiment_group']).agg({
    'conversions': 'sum'
}).reset_index()

# Pivot for easier calculation
region_pivot = region_group.pivot(index='geo_region', columns='experiment_group', values='conversions').reset_index()

# Calculate Lift
region_pivot['lift'] = (region_pivot['Treatment'] - region_pivot['Control']) / region_pivot['Control']

# Visualization: Incrementality Lift by Region
plt.figure(figsize=(8,5))
plt.bar(region_pivot['geo_region'], region_pivot['lift'], color='orange')
plt.ylabel('Lift (%)')
plt.title('Incrementality Lift by Region')
plt.show()
```

4. Attribution Models: Last-Touch vs Linear vs Incrementality
```# Simulate customer journeys
import numpy as np
import pandas as pd
from collections import defaultdict

np.random.seed(42)
channels = ['Search', 'Social', 'Display', 'Email', 'Affiliate']
customers = []
for cust_id in range(1, 1001):
    num_touches = np.random.randint(1, 5)
    touch_sequence = np.random.choice(channels, size=num_touches, replace=False)
    conversion = np.random.choice([1, 0], p=[0.7, 0.3])
    customers.append([cust_id, list(touch_sequence), conversion])
mta_df = pd.DataFrame(customers, columns=['customer_id','touchpoints','conversion'])

# Last-Touch Attribution
last_touch = (
    mta_df[mta_df['conversion'] == 1]
    .apply(lambda row: row['touchpoints'][-1], axis=1)
    .value_counts()
    .reset_index()
)
last_touch.columns = ['channel', 'last_touch_conversions']

# Linear Attribution
linear_credit = defaultdict(float)
for _, row in mta_df.iterrows():
    if row['conversion'] == 1:
        credit = 1 / len(row['touchpoints'])
        for ch in row['touchpoints']:
            linear_credit[ch] += credit
linear_df = pd.DataFrame(list(linear_credit.items()), columns=['channel', 'linear_conversions'])

# Incrementality-based (from dataset)
incrementality_df = df.groupby('channel')['incremental_conversions'].sum().reset_index()
incrementality_df.columns = ['channel', 'incrementality_conversions']

# Merge all
mta_compare = last_touch.merge(linear_df, on='channel').merge(incrementality_df, on='channel')

# Visualization: Attribution Models Comparison
mta_compare.set_index('channel')[['last_touch_conversions', 'linear_conversions', 'incrementality_conversions']].plot(kind='bar', figsize=(10,6))
plt.title('Attribution Models: Last-Touch vs Linear vs Incrementality')
plt.ylabel('Conversions')
plt.show()
```


 5. Budget Changes: Recommended vs Current
```import matplotlib.pyplot as plt
import numpy as np

# Prepare data
budget_diff = channel_summary['recommended_spend'] - channel_summary['spend']
channels = channel_summary['channel']

plt.figure(figsize=(10,6))
plt.bar(channels, budget_diff, color=['green' if x > 0 else 'red' for x in budget_diff])
plt.axhline(0, color='black', linewidth=1)
plt.title('Budget Changes: Recommended vs Current')
plt.ylabel('Budget Difference ($)')
plt.show()
```
________________________________________
**📈 Key Insights & Business Impact**

-Email Marketing Severely Underfunded: Highest incremental ROI (1.71x) but only 3.2% of budget

-TV Budget Massively Inflated: Lowest incremental ROI (0.21x) consuming 43.7% of budget

-Attribution Bias Universal: All channels show 52-61% bias in traditional measurement

-EU Outperforms All Regions: 143% lift vs 127% (US) and 107% (APAC)

-Multi-Touch vs Single-Touch: Customer journey analysis reveals attribution complexity

**Business Recommendations**
-Immediate: Reallocate $13.5M from TV to Email/Affiliate channels

-Regional: Increase EU investment given superior 143% lift performance

-Measurement: Replace last-touch attribution with incrementality-based models

-Testing: Expand geo-lift experiments to validate channel interactions
________________________________________
**Technical Implementation**
**Core Technologies**
-Python: Pandas, NumPy, Matplotlib, Seaborn

-Stats: Geo-Lift, t-tests, bootstrap

-Modeling: Linear Regression for MMM

-Visualization: Traditional vs Incremental ROI, Budget Allocation: Current vs Recommended, Incrementality Lift by Region, Attribution Models: Last-Touch vs Linear vs Incrementality, Budget Changes: Recommended vs Current

________________________________________
**Why This Matters for Modern Marketing**
This analysis demonstrates the fundamental shift happening in marketing measurement:

Old Way: Correlation-based attribution, wasted spend, gut-feel decisions

New Way: Causal inference, incrementality focus, scientific optimization

Companies implementing incrementality measurement see:

-20-40% improvement in marketing efficiency

-Millions saved from eliminating non-incremental spend

-Data-driven culture replacing opinion-based budget fights
________________________________________
**Future Enhancements**

-Real-Time Monitoring: Live incrementality dashboards

-Advanced MMM: Saturation curves and adstock modeling

-Synthetic Controls: Enhanced causal inference methods

-Cross-Channel Attribution: Interaction effect measurement
________________________________________
**Contact & Collaboration**

Built by Advait Athalye

📧 Email: advaiitathalye@gmail.com

🔗 LinkedIn: [linkedin.com/in/advait-athalye](https://www.linkedin.com/in/advaitathalye)


