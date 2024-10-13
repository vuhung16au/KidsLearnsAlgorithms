import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import calendar
import random
from mpl_toolkits.mplot3d import Axes3D

# Function to generate test data
def generate_test_data(num_days=365, start_date='2024-01-01'):
    # Create date range for the year
    date_range = pd.date_range(start=start_date, periods=num_days)
    # Random number of commits for each day
    commit_counts = [random.randint(0, 20) for _ in range(num_days)]
    # Create a DataFrame with the data
    data = pd.DataFrame({'Date': date_range, 'Commits': commit_counts})
    return data

# Generate test data
data = generate_test_data(90, '2024-01-01')
print(data.head())

# Generate test data for one year
data = generate_test_data()

# Add additional columns for day, month, and weekday
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Weekday'] = data['Date'].dt.weekday  # Monday=0, Sunday=6

# Pivot table to create the heatmap matrix by weekday and month
heatmap_data = data.pivot_table(index='Weekday', columns='Month', values='Commits', aggfunc='sum', fill_value=0)

# Create weekday labels (Mon to Sun)
weekday_labels = [calendar.day_name[i] for i in range(7)]
month_labels = [calendar.month_abbr[i] for i in range(1, 13)]

# Plot the 2D heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap="Greens", annot=True, fmt="d", cbar=True, xticklabels=month_labels, yticklabels=weekday_labels)
plt.title("GitHub-style Contribution Heatmap (Commits per Day)")
plt.xlabel("Month")
plt.ylabel("Weekday")
plt.show()

# 3D Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Prepare data for 3D plotting
x_data, y_data = np.meshgrid(range(1, 13), range(7))
x_data = x_data.flatten()
y_data = y_data.flatten()
z_data = heatmap_data.values.flatten()

# Plot the bars
ax.bar3d(x_data, y_data, np.zeros_like(z_data), 1, 1, z_data, shade=True, cmap='Greens')

# Set labels
ax.set_xlabel('Month')
ax.set_ylabel('Weekday')
ax.set_zlabel('Commits')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.set_yticks(range(7))
ax.set_yticklabels(weekday_labels)
ax.set_title('3D GitHub-style Contribution Heatmap')

plt.show()

""" 
Prompt: 

Help me create a heatmap to show github contribution based on the numbers of commits over the days in a year

Input: 
- List of dates
- Number of commits in the days 

For example: 

2024/10/10, 30 (30 commits)
2024/10//11, 2 (2 commits)

Output: 
A heatmap like github.com to show the numbers of commits 
Show the  heatmap for number of commits for weekdays (Mon, Tue... Sat, Sun)
Show the  heatmap for number of commits for months (Jan, Feb... Nov, Dec) 

Generate test data 
Generate test code to generate test data for 365 days

"""
