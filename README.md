# ML_BaseballCLF_Wdocker

Baseball has been a topic of interest for many Americans and is one of America's favorite pastimes.
Even more than baseball, baseball betting is one of the larger sectors in the sports betting industry. 
In this project, I created a machine learning model to attempt to predict whether the home team will win the game.
To accomplish this I showcase feature inspection and engineering using both python and SQL from a pre-constructed database of Baseball games from 
the years 2007 to 2012. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To install the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone --branch dev https://github.com/krastegar/BDA_602.git
    ```
2. Navigate to the project directory:
    ```sh
    cd BDA_602/
    ```
3. Build the Docker container:
    ```sh
    docker-compose build
    ```

## Usage

To run the project, use the following command:
```sh
docker-compose up
```
## Features
This project includes reproducible analysis that contains a machine learning model and is containerized. Some key features of the analysis are:

### Feature Engineering
All features used to train subsequent machine learning models were engineered in SQL. Some feature engineering operations resulted in producing null values due to divisions by 0. These null values were treated as 0s because they were null due to division by 0. The list of features is shown below:

-Away_RelPitch_Ratio
-Home_RelPitch_Ratio
-Away_NumStrikes
-Home_NumStrikes
-Away_StrikeRatio
-Home_StrikeRatio
-Away_BattingAvg
-Home_BattingAvg
-Away_HR_Ratio
-Home_HR_Ratio
-Away_OBP
-Home_OBP
-Diff_ReliefPitch
-Diff_NumStrikes
-Diff_Strk_Wlk_Ratio
-Diff_BA
-Diff_HR_ratio
-Diff_OBP

### Inspecting Features
-p-values and Logistic Regression
-Since we are predicting a binary outcome "home team wins" (0 or 1) and all features are continuous variables, a quick check on all features using logistic regression was performed. The importance of these features was summarized by a bar chart to determine which ones would be the most predictive in the model.

The top two predictive features are Away_team_batting_average and home_team_HR_ratio, while the least predictive feature is the difference between the teams' BAs.

### Difference of Means
To see any trends in the data, the distribution of the data was analyzed, observing the difference between the population mean and the sample means in each bin.

### Combination of Features
Using a brute force model, trends in the combinations of features were observed by analyzing the squared differences of means between sample and population in 2 dimensions.

### Pearson's Correlation
A correlation test on all features was conducted to determine if any features needed to be dropped. Most features did not correlate well, which is beneficial for feeding these features into a logistic regression model.

### Results
#### Models and Training
Two different models were trained with an 80/20 training/testing split:

GradientBoostingClassifier
LogisticRegressor

#### Model Evaluation
The models were evaluated by looking at the ROC curve. The GradientBoostingClassifier had a slight edge over the LogisticRegressor model with an AUC score of 0.64.

## Contributing
There are no specific guidelines for contributing to this project.

## License
There is no license specified for this project.

## Contact
If you have any questions or feedback, feel free to contact me at krastegar0@gmail.com.

## Project Wiki Link
Results of project can be found at this link

Project link: https://github.com/krastegar/ML_BaseballCLF_Wdocker/wiki
