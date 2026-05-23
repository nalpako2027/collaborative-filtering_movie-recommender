# 🚀 Sci‑Fi Movie Recommendation System Using KNN Collaborative Filtering

## Project Overview

This project develops a memory-based collaborative filtering recommendation system using the MovieLens 32M dataset. The analysis focuses specifically on Sci‑Fi movie preferences to reduce sparsity and improve neighborhood similarity in K-Nearest Neighbors (KNN)-based recommendation modeling. The project demonstrates an end-to-end machine learning workflow including preprocessing, user-profile construction, hyperparameter tuning, evaluation and interpretation of the findings. 

### Objectives

- **Business Case & Objective:** In the entertainment streaming industry, personalizing content delivery is critical to maximizing user retention and platform engagement. This project builds an end-to-end predictive pipeline using a $K$-Nearest Neighbors (KNN) Regression framework to accurately forecast user movie ratings, enabling data-driven, automated content recommendations.
- Build a KNN-based movie recommendation system.
- Data engineering and manipulation.
- Reduce sparsity by restricting the analysis on a specific genre (i.e., Sci‑Fi movie profiles).
- Compare similarity metrics and neighborhood sizes.
- Machine learning and validation
- Demonstrate practical recommender-system methodology for industry-oriented analytics.

### Dataset

The project uses the MovieLens dataset (Harper & Konstan, 2015) provided by GroupLens Research. The dataset contains user ratings, movie metadata, genres, and related identifiers. The analysis primarily uses user ratings and movie genres.
(F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872)
Original data was downloaded from https://grouplens.org/datasets/movielens/32m/

## Methodology

- Merged ratings and movie metadata tables.
- Deterministic sampling: Filtered Sci‑Fi movies using genre labels. To address the challenge of high data sparsity common in collaborative filtering datasets, the scope of this project was narrowed strictly to the 'Sci-Fi' genre. Including all genres introduces a vast number of unrated items per user, resulting in a sparse matrix that can degrade model performance. The method of filtering for Sci-Fi increases data density and ensure that the model trains on a concentrated pool of overlapping user preferences, thereby improving the reliability of the recommendations while optimizing computational efficiency.
- Selected active users with sufficient rating counts.
- Constructed a user–movie rating matrix.
- Applied KNN regression for collaborative filtering.
- Hyperparameter Tuning: Executed cross-validation loops to systematically evaluate optimal k-values and similarity metrics.

## Key Findings & Model Performance

### 📈 Model Evaluation Metrics
* **Optimal Hyperparameter ($k$):** `38` (The smallest cluster size that minimizes prediction error).
* **Minimum Mean Squared Error (MSE):** `0.6643`
* **Root Mean Squared Error (RMSE):** `0.816` 
* **Mean Absolute Error (MAE):** `0.627`
* **Coefficient of Determination ($R^2$ Score):** `0.1545`

### 💡 Data Science Insights & Interpretation
1. **High Prediction Accuracy:** The **MAE of 0.62** demonstrates that, on average, the KNN model’s predictions are within roughly **0.6 stars** of the user's actual rating on a standard 1-to-5 star scale. This provides a highly reliable baseline for content serving.
2. **Optimized Neighborhood Size:** Isolating **$k=38$** strikes the perfect bias-variance tradeoff. Smaller neighborhood sizes caused the model to overfit to localized noise, while larger values diluted the unique preferences of Sci-Fi sub-genres.
3. **Behavioral Complexity ($R^2$):** The $R^2$ score of 0.1545 successfully captures over 15% of the variance in highly subjective user viewing habits. In collaborative filtering and human behavioral modeling, this represents a statistically significant signal for a baseline distance-based algorithm.
4. Sci‑Fi filtering reduced sparsity and improved neighborhood quality as evidenced by lower mean squared error (MSE).
5. Manhattan similarity metric produced slightly better results than Cosine and Euclidean distance. This can be attributed to mean replacement of missing ratings by 3.

### Requirements
- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- Jupyter Notebook

### Further Analytical Recommendations
- Compare user-based and item-based collaborative filtering.
- Incorporate movie metadata and tags.
- Evaluate ranking-based recommendation metrics.
