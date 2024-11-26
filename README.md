# AIPI510_Final_Project

# YouTube Dataset Analysis

## Executive Summary
This project aims to collect and analyze data, including titles, tags, thumbnails, and view counts, from a specific YouTube channel. The EDA focuses on two main parts: creating word clouds and conducting sentiment analysis on comments, and extracting colors from thumbnails to analyze the correlation between thumbnail colors and view counts. Potential applications include optimizing video content strategies, enhancing viewer engagement, and conducting market research.

## Data Description
The data used in this project is sourced from the YouTube API. It includes:
- Video Details: Titles, publish dates, tags, view counts, like counts, comment counts, and thumbnail URLs.
- Comments: First 200 comments on the latest video.
- Thumbnails: Images associated with each video.

## Power Analysis
The power analysis is conducted to determine the required sample size for detecting a medium effect size (d=0.5) with a significance level (α=0.5) and power (β=0.8) The calculated sample size ensures reliable statistical analysis.

## Exploratory Data Analysis (EDA)

### Sentiment Analysis
- Comments for selected videos were analyzed for sentiment by using TextBlob, categorizing them as positive, negative, or neutral.
- This provides insights into audience reactions.
![Sentiment Analysis](https://imgur.com/a/SPgAUSU)

### Word Cloud Generation
- A word cloud was created from video comments, visualizing frequently used terms.
- This would reveal common themes or topics of discussion.

### Thumbnail Analysis
- Dominant colors were extracted from video thumbnails, and contrast between colors was calculated.

### Correlation Analysis
- The analysis explored correlations between thumbnail features (colors and contrast) and view counts, which could indicate potential relationships between visual elements and video popularity.
- This data can be used to explore potential correlations between thumbnail aesthetics and video performance.

## Link to Data
[Kaggle](https://www.kaggle.com/datasets/kelllychen/youtuber-data-mrbeast/data)


## Ethics Statement
### Privacy and Data Usage
- No personal or sensitive data is collected.
- Data is used solely for research and analysis purposes.
- Results are reported in aggregate form without identifying individual users or channels.

### Bias Awareness
- The project recognizes that the dataset may contain inherent biases, including selection bias from the chosen channels.

## Open Source License
### This project is licensed under the CC BY-NC-SA 4.0 License.
