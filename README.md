# NewChic E-Commerce: Identifying Premier Products and Optimal Categories for Strategic Investment

> A clustering-and-classification study of NewChic.com product data that identifies the top 10 products and the strongest category among women, men, house, and shoes, then turns the findings into investment recommendations.
>
> **Team data-mining assignment; the author served as project leader of a six-member team.**

`Python` · `pandas` · `scikit-learn` · `K-means` · `Agglomerative (hierarchical) clustering` · `KNN` · `Gaussian Naive Bayes`

## Problem
NewChic.com lists products across many categories, and the business needs to know which individual products are its strongest performers ("top 10") and which of the women, men, house, and shoes categories is the best target for investment. The project defines "top" and "best" in terms of current price, discount rate, and like count (a proxy for popularity), then mines the product data to answer both questions and recommend where to allocate resources.

## Approach
1. **Data preparation** — Consolidated nine CSV export files (49,631 product records) and restricted the analysis to the women, men, house, and shoes categories. Kept the numerical drivers `current_price`, `discount`, and `likes_count`; dropped `raw_price` because it was highly correlated with `current_price`. Removed invalid records (238 products with a raw price of 0, 2 with a 100% discount) and duplicates, leaving **49,221 products**, and standardized the features with `StandardScaler`.
2. **Clustering** — Ran K-means (k = 4, chosen with the elbow method) and Agglomerative/hierarchical clustering (Ward linkage, Euclidean distance) at 3, 4, and 6 clusters, then compared the cluster profiles. K-means was preferred as the more efficient option that still produced clusters with distinct characteristics.
3. **Product scoring and labeling** — Rank-scored each product on `likes_count`, `current_price`, and `discount`, combining the three rankings with weights of **50% / 30% / 20%** into a composite score, then split the ranked products into four balanced label groups (0–3) for supervised learning.
4. **Classification** — Split the data 60/20/20 into train/validation/test sets, benchmarked eight classifiers with ShuffleSplit cross-validation, and selected K-Nearest Neighbors (tuned with `GridSearchCV`) and Gaussian Naive Bayes. Evaluated with accuracy, confusion matrices, a precision/recall/F1 report, and a t-test.
5. **Recommendations** — Combined the top-scoring products and per-category scores into concrete investment guidance.

## Results
- **Cleaned analytical dataset:** 49,221 products across four categories, described by three standardized features. No missing values were found; 238 zero-raw-price records, 2 full-discount records, and duplicates were removed.
- **K-means (k = 4):** cluster sizes of 7,842 / 21,316 / 509 / 19,554; the small 509-product cluster is characterized by a very high mean like count (~4,640), separating premium/popular products from the rest.
- **Hierarchical clustering (Ward):** modeled at 3 clusters (12,664 / 35,226 / 1,331), 4 clusters, and 6 clusters; results were consistent with K-means, and K-means was chosen for efficiency.
- **Category scores** from the weighted scoring model (higher = stronger):

  | Category | Total score |
  |----------|-------------|
  | Shoes    | 12,614,721.2 |
  | Women    | 10,918,689.2 |
  | Men      | 7,738,681.7 |
  | House    | 6,636,527.2 |

- **Classifier selection:** Of eight benchmarked classifiers, tree-based models reached ~99% test accuracy but a perfect 1.0 training accuracy and were excluded as overfit; KNN and Gaussian Naive Bayes were selected.
- **KNN (best k = 5):** test accuracy **95.66%**, validation accuracy **96.17%**, and a 10-fold cross-validation mean of **97.68%** (range 96.75%–98.14%). On the test set (n = 9,845) it scored ~0.96 precision/recall/F1.
- **Gaussian Naive Bayes:** test accuracy **84.39%**, validation accuracy **83.84%**, 10-fold CV mean **86.34%**.
- **KNN outperformed Gaussian Naive Bayes**, a difference confirmed as statistically significant by a two-sample t-test (t = -51.43, p ≈ 0.0000).
- **Top 10 products:** 9 of the 10 came from the **shoes** category and 1 from **men**; all shared relatively high current prices, moderate discounts, and high like counts, and all fell in the same product cluster.
- **Best category and recommendation:** **Shoes** was identified as the best category (highest total score, 90% of the top 10, and a broad range of subcategories). The project recommends investing more in **shoes and women**, avoiding low-score products (low likes, low price, high discount), and deprioritizing the **house** category.

## Repository structure
```
├── Project.ipynb   # end-to-end analysis: cleaning, clustering, scoring model, classification
└── README.md
```
Note: the notebook loads NewChic CSV export files from its working directory; those data files are not included in the repository.

## Tech
- **Language / environment:** Python, Jupyter Notebook
- **Data & math:** pandas, NumPy, SciPy (`scipy.cluster.hierarchy`)
- **Machine learning (scikit-learn):** `KMeans`, `AgglomerativeClustering`, `KNeighborsClassifier`, `GaussianNB`, `GridSearchCV`, `ShuffleSplit`, `cross_validate`, `train_test_split`, `StandardScaler`, `metrics`
- **Visualization:** Matplotlib, seaborn

## Team & role
Completed as a team assignment; the author served as **project leader of a six-member team**.
