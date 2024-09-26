import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Load the data
data = pd.read_csv('instagram_data.csv')

""" OUTLIER DETECTION -- remove posts with unusually low engagement.
Define low engagement as low ratio of likes to followers (e.g. poster
has lots of followers but the post gets few likes) """

# Create likes-to-followers ratio
data['likes_to_followers_ratio'] = data['likes'] / data['follower_count_at_t']

# Calculate 5th percentile threshold for likes-to-followers ratio
threshold = data['likes_to_followers_ratio'].quantile(0.05)

# Filter out the bottom 5%
filtered_data = data[data['likes_to_followers_ratio'] >= threshold].copy()

""" FEATURE ENGINEERING -- make timestamps more meaningful by extracting
    hours of the day and days of the week """

# First cast 't' column to ensure no conflicting types
filtered_data['t'] = filtered_data['t'].astype('int64')

# Convert the timestamp from Unix format to a datetime object
filtered_data['t'] = pd.to_datetime(filtered_data['t'], unit='s')

# Extract hour of the day using .loc
filtered_data.loc[:, 'hour'] = filtered_data['t'].dt.hour

# Extract day of the week using .loc (0 = Monday, 6 = Sunday)
filtered_data.loc[:, 'day_of_week'] = filtered_data['t'].dt.dayofweek

# Extract month using .loc
filtered_data.loc[:, 'month'] = filtered_data['t'].dt.month

# Export the filtered data with feature engineering to a CSV file
filtered_data.to_csv('filtered_instagram_data_with_features.csv', index=False)

""" REGRESSION -- random forest model performs better than linear model here so I'll use that """

# Define features (excluding 'likes', which is the target)
features = ['no_of_comments', 'follower_count_at_t', 'hour', 'day_of_week', 'month']
X = filtered_data[features]
y = filtered_data['likes']

# Split data into train and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model using RMSE
rmse_rf = root_mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest RMSE: {rmse_rf}")

# Check feature importance from the Random Forest model
importances = rf_model.feature_importances_
feature_names = X.columns

# Print feature importances
for feature, importance in zip(feature_names, importances):
    print(f"Feature: {feature}, Importance: {importance}")


# Scatter plot: Number of Comments vs Likes
plt.figure(figsize=(8, 6))
sns.scatterplot(x='no_of_comments', y='likes', data=filtered_data)
plt.title('Number of Comments vs Likes')
plt.xlabel('Number of Comments')
plt.ylabel('Number of Likes')
plt.savefig('comments_vs_likes.png')
plt.show()

# Scatter plot: Follower Count vs Likes
plt.figure(figsize=(8, 6))
sns.scatterplot(x='follower_count_at_t', y='likes', data=filtered_data)
plt.title('Follower Count vs Likes')
plt.xlabel('Follower Count')
plt.ylabel('Number of Likes')
plt.savefig('followers_vs_likes.png')
plt.show()

# Bar plot: Hour of Posting vs Average Likes
plt.figure(figsize=(8, 6))
hourly_likes = filtered_data.groupby('hour')['likes'].mean()
hourly_likes.plot(kind='bar')
plt.title('Hour of Posting vs Average Likes')
plt.xlabel('Hour of Day')
plt.ylabel('Average Number of Likes')
plt.savefig('hour_vs_likes.png')
plt.show()

# Bar plot: Day of the Week vs Average Likes
plt.figure(figsize=(8, 6))
day_likes = filtered_data.groupby('day_of_week')['likes'].mean()
day_likes.plot(kind='bar')
plt.title('Day of the Week vs Average Likes')
plt.xlabel('Day of Week (0 = Monday, 6 = Sunday)')
plt.ylabel('Average Number of Likes')
plt.savefig('day_vs_likes.png')
plt.show()

# Histogram of likes
plt.figure(figsize=(8, 6))
plt.hist(filtered_data['likes'], bins=50)
plt.title('Distribution of Likes')
plt.xlabel('Number of Likes')
plt.ylabel('Frequency')
plt.savefig('likes_distribution.png')
plt.show()