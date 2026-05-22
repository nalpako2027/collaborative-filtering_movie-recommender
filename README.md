# Sci‑Fi Movie Recommendation System Using KNN Collaborative Filtering

## Project Overview

This project develops a memory-based collaborative filtering recommendation system using the MovieLens dataset. The analysis focuses specifically on Sci‑Fi movie preferences to reduce sparsity and improve neighborhood similarity in K-Nearest Neighbors (KNN)-based recommendation modeling. The project demonstrates an end-to-end machine learning workflow including preprocessing, user-profile construction, hyperparameter tuning, evaluation and interpretation of the findings.

### Objectives

- Build a KNN-based movie recommendation system.
- Reduce sparsity by restricting the analysis on a specific genre (i.e., Sci‑Fi movie profiles).
- Compare similarity metrics and neighborhood sizes.
- Evaluate model performance using RMSE and MAE.
- Demonstrate practical recommender-system methodology for industry-oriented analytics.

### Dataset

The project uses the MovieLens dataset (Harper & Konstan, 2015) provided by GroupLens Research. The dataset contains user ratings, movie metadata, genres, and related identifiers. The analysis primarily uses user ratings and movie genres.
(F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872)

## Methodology

- Merged ratings and movie metadata tables.
- Filtered Sci‑Fi movies using genre labels. To address the challenge of high data sparsity common in collaborative filtering datasets, the scope of this project was narrowed strictly to the 'Sci-Fi' genre. Including all genres introduces a vast number of unrated items per user, resulting in a sparse matrix that can degrade model performance. The method of filtering for Sci-Fi increases data density and ensure that the model trains on a concentrated pool of overlapping user preferences, thereby improving the reliability of the recommendations while optimizing computational efficiency.
- Selected active users with sufficient rating counts.
- Constructed a user–movie rating matrix.
- Applied KNN regression for collaborative filtering.
- Tuned k-values and evaluated similarity metrics.

## Key Findings
- Moderate k-values (e.g., k=25) produced better performance than smaller neighborhoods and the performance was similar to larger k-values (k>=25).
- Sci‑Fi filtering reduced sparsity and improved neighborhood quality as evidenced by lower mean squared error (MSE).
- Manhattan similarity metric produced slightly better results than Cosine and Euclidean distance. This can be attributed to mean replacement of missing ratings by 3.
- RMSE values stabilized around 0.81 with MAE around 0.63.

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
