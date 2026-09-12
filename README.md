![License](https://img.shields.io/github/license/nalpako2027/collaborative-filtering_movie-recommender?style=for-the-badge)
![Last Commit](https://img.shields.io/github/last-commit/nalpako2027/collaborative-filtering_movie-recommender?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/orhan-kaplan-phd-5a4a84212/)

# 🚀 Sci‑Fi Movie Recommendation System Using KNN Collaborative Filtering 
## (Status: 🏗️ The Project Under Construction, adding a case study and SVD/NMF analysis)

## 📝 Summary
This project develops a memory-based collaborative filtering recommendation system using the MovieLens 32M dataset. The analysis focuses specifically on Sci‑Fi movie preferences to reduce sparsity and improve neighborhood similarity in K-Nearest Neighbors (KNN)-based recommendation modeling. The project demonstrates an end-to-end machine learning workflow including preprocessing, user-profile construction, hyperparameter tuning, evaluation and interpretation of the findings. 

## 🎯 Objectives

- **Business Case & Objective:** Streaming platforms live or die on personalization. This project builds a *KNN*-based recommendation engine that predicts user ratings for Sci-Fi movies, giving platforms a data-driven way to boost engagement and retention through targeted content delivery.
- Build a KNN-based movie recommendation system.
- Data engineering and manipulation.
- Reduce sparsity by restricting the analysis on a specific genre (i.e., Sci‑Fi movie profiles).
- Compare similarity metrics and neighborhood sizes.
- Machine learning and validation
- Demonstrate practical recommender-system methodology for industry-oriented analytics.



### 📂 Dataset

The project uses the MovieLens dataset (Harper & Konstan, 2015) provided by GroupLens Research. The dataset contains user ratings, movie metadata, genres, and related identifiers. The analysis primarily uses user ratings and movie genres.
(F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872)
Original data was downloaded from https://grouplens.org/datasets/movielens/32m/

## 🔧 Methodology

- Merged ratings and movie metadata tables.
- Filtered Sci‑Fi movies using genre labels. To address the challenge of high data sparsity common in collaborative filtering datasets, the scope of this project was narrowed strictly to the 'Sci-Fi' genre. Including all genres introduces a vast number of unrated items per user, resulting in a sparse matrix that can degrade model performance. However, this genre restriction also limits the model's ability to leverage users' cross-genre preferences and changes the effective user population toward users with stronger Sci-Fi rating activity. Therefore, the findings should be interpreted as applying to Sci-Fi-focused recommendation rather than general-purpose movie recommendation. The method of filtering for Sci-Fi genre increases data density and ensure that the model trains on a concentrated pool of overlapping user preferences, thereby improving the reliability of the recommendations while optimizing computational efficiency.
- Selected active users with sufficient rating counts.
- Constructed a user–movie rating matrix.
- Applied KNN regression for collaborative filtering.
- Hyperparameter Tuning: Executed cross-validation loops to systematically evaluate optimal k-values and similarity metrics.

## 📊 Key Findings & Model Performance

### 📈 Model Evaluation Metrics
* **Optimal Hyperparameter ($k$):** `38` (The smallest cluster size that minimizes prediction error).
* **Minimum Mean Squared Error (MSE):** `0.6643`
* **Root Mean Squared Error (RMSE):** `0.816` 
* **Mean Absolute Error (MAE):** `0.627`
* **Coefficient of Determination ($R^2$ Score):** `0.1545`

## ⚙️ Requirements  
![Python](https://img.shields.io/badge/Python-3.11-blue)  
![pandas](https://img.shields.io/badge/Pandas-3.11-blue)  
![NumPy](https://img.shields.io/badge/NumPy-3.11-blue)
![matplotlib](https://img.shields.io/badge/Matplotlib-3.11-blue)
![scikit-learn](https://img.shields.io/badge/numpy-3.11-blue)
![Jupyter Notebook](https://img.shields.io/badge/numpy-3.11-blue)
![NumPy](https://img.shields.io/badge/numpy-3.11-blue)



### 🧩 Business Questions & Key Insights
1. **High Prediction Accuracy:** The **MAE of 0.62** demonstrates that, on average, the KNN model’s predictions are within roughly **0.6 stars** of the user's actual rating on a standard 1-to-5 star scale. The model captures a modest but real signal typical of memory-based collaborative filtering.
2. **Optimized Neighborhood Size:** Isolating **$k=38$** strikes the best bias-variance tradeoff. Smaller neighborhood sizes caused the model to overfit to localized noise, while larger values diluted the unique preferences of Sci-Fi sub-genres.
3. **Behavioral Complexity ($R^2$):** The $R^2$ score of 0.1545 successfully captures over 15% of the variance in highly subjective user viewing habits. In collaborative filtering and human behavioral modeling, this represents a modest signal for a baseline distance-based algorithm.
4. Sci-Fi filtering reduced the sparsity of the user–movie matrix and provided a more concentrated set of overlapping user preferences for neighborhood-based modeling.
5. Manhattan similarity metric produced slightly better results than Cosine and Euclidean distance. This can be attributed to mean replacement of missing ratings by 3.


### Further Analytical Recommendations


## ▶️ How to Reproduce
Original data was downloaded from https://grouplens.org/datasets/movielens/32m/

Suggested Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

🗂️ Project Structure


## ⚠️ Limitations & Further Recommendations  
- Compare user-based and item-based collaborative filtering.
- Incorporate movie metadata and tags.
- Evaluate ranking-based recommendation metrics.

## 📚 References

## License

## 👤 Author/Attribution
If you use this project in research, publications, presentations, or other work, please cite or acknowledge:

Orhan Kaplan (2026). *Sci‑Fi Movie Recommendation System for "Cold-Start" Users: Collaborative Filtering vs. Matrix Factorization*.
https://github.com/nalpako2027/collaborative-filtering_movie-recommender.git

