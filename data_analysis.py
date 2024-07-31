import pandas as pd  # Importing pandas for data manipulation and analysis
import numpy as np  # Importing numpy for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for data visualization
import seaborn as sns  # Importing seaborn for advanced data visualization

# Load the Titanic dataset from seaborn's built-in datasets
titanic = sns.load_dataset('titanic')

# Display the first few rows of the dataset to understand its structure
print(titanic.head())

# Display basic information about the dataset including column names and data types
print(titanic.info())

# Display summary statistics for numerical columns in the dataset
print(titanic.describe())

# Check for missing values in the dataset
print(titanic.isnull().sum())

# Handle missing values in 'age' by filling with the median age
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# Handle missing values in 'embarked' by filling with the mode (most frequent value)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# Verify that there are no more missing values in 'age' and 'embarked'
print(titanic.isnull().sum())

# Convert the 'sex' column from categorical to numerical (male: 0, female: 1)
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

# Convert the 'embarked' column from categorical to numerical (C: 0, Q: 1, S: 2)
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Display the first few rows to verify the transformations
print(titanic.head())

# Create a histogram to visualize the age distribution
plt.figure(figsize=(10, 6))  # Set the figure size
sns.histplot(titanic['age'], kde=True)  # Plot the histogram with a KDE (Kernel Density Estimate)
plt.title('Age Distribution')  # Add a title to the plot
plt.show()  # Display the plot

# Create a bar plot to visualize survival rate by class
plt.figure(figsize=(10, 6))  # Set the figure size
sns.barplot(x='class', y='survived', data=titanic)  # Plot the bar plot with class on x-axis and survival rate on y-axis
plt.title('Survival Rate by Class')  # Add a title to the plot
plt.show()  # Display the plot

# Create a bar plot to visualize survival rate by sex
plt.figure(figsize=(10, 6))  # Set the figure size
sns.barplot(x='sex', y='survived', data=titanic)  # Plot the bar plot with sex on x-axis and survival rate on y-axis
plt.title('Survival Rate by Sex')  # Add a title to the plot
plt.show()  # Display the plot

# Create a pair plot to visualize relationships between multiple variables
sns.pairplot(titanic[['age', 'fare', 'survived', 'pclass', 'sex', 'embarked']])  # Select specific columns for the pair plot
plt.show()  # Display the plot

# Calculate and print the overall survival rate
survival_rate = titanic['survived'].mean()  # Calculate the mean survival rate
print(f'Survival Rate: {survival_rate:.2%}')  # Print the survival rate formatted as a percentage

# Calculate and print the survival rate by sex
survival_rate_by_sex = titanic.groupby('sex')['survived'].mean()  # Group by sex and calculate the mean survival rate for each group
print('Survival Rate by Sex:')  # Print a heading
print(survival_rate_by_sex)  # Print the survival rates by sex

# Calculate and print the survival rate by class
survival_rate_by_class = titanic.groupby('class')['survived'].mean()  # Group by class and calculate the mean survival rate for each group
print('Survival Rate by Class:')  # Print a heading
print(survival_rate_by_class)  # Print the survival rates by class



