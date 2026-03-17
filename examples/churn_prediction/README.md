# Predict Customer Churn

Playground Series - Season 6 Episode 3

![](https://www.kaggle.com/competitions/125197/images/header)

## Predict Customer Churn

**Submit Prediction**

[](https://www.kaggle.com/competitions/playground-series-s6e3/overview)[](https://www.kaggle.com/competitions/playground-series-s6e3/data)[](https://www.kaggle.com/competitions/playground-series-s6e3/code)[](https://www.kaggle.com/competitions/playground-series-s6e3/models)[](https://www.kaggle.com/competitions/playground-series-s6e3/discussion)[](https://www.kaggle.com/competitions/playground-series-s6e3/leaderboard)[](https://www.kaggle.com/competitions/playground-series-s6e3/rules)[](https://www.kaggle.com/competitions/playground-series-s6e3/team)[](https://www.kaggle.com/competitions/playground-series-s6e3/submissions)

## Overview

**Welcome to the 2026 Kaggle Playground Series!** We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Predict the likelihood of customer churn.

### Evaluation

The evaluation section describes how submissions will be scored and how participants should format their submissions. You don't need a sub-title at the top; the page title appears above. Below is an example of a typical evaluation page.

---

Submissions are evaluated on [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.

## Submission File

For each id in the test set, you must predict a probability for the `Churn` variable. The file should contain a header and have the following format:

```apache
id,Churn
594194,0.1
594195,0.3
594196,0.2
etc.
```

### Timeline

* **Start Date** - March 1, 2026
* **Entry Deadline** - Same as the Final Submission Deadline
* **Team Merger Deadline** - Same as the Final Submission Deadline
* **Final Submission Deadline** - March 31, 2026

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!


### Dataset Description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the [customer churn prediction dataset](https://www.kaggle.com/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm). Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

## Files

* **train.csv** - the training set, where `Churn` is the target
* **test.csv** - the test set
* **sample_submission.csv** - a sample submission file in the correct format

### Prizes

* 1st Place - Choice of Kaggle merchandise
* 2nd Place - Choice of Kaggle merchandise
* 3rd Place - Choice of Kaggle merchandise

**Please note:** In order to encourage more participation from beginners, Kaggle merchandise will only be awarded once per person in this series. If a person has previously won, we'll skip to the next team

### Citation

Yao Yan, Walter Reade, Elizabeth Park. Predict Customer Churn. https://kaggle.com/competitions/playground-series-s6e3, 2026. Kaggle.
