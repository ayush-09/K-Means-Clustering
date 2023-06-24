# Unsupervised Learning Algorithm

This project implements the K-means++ clustering algorithm for unsupervised learning using Python. The dataset used is the Salary dataset from Kaggle, which contains information about years of experience and corresponding salaries.

## Getting Started

1. Clone the repository or download the project files.

2. Open the Jupyter Notebook file `Unsupervised_Learning.ipynb` using Jupyter Notebook or Google Colab.

3. Run the code cells in the notebook sequentially to execute the algorithm and generate the results.

## Dataset

The Salary dataset (`salary.csv`) is used for this project. It contains two columns: "YearsExperience" and "Salary". The data is visualized using scatter plots to gain insights into the data distribution.<br>
Kaggle:  <a href="https://www.kaggle.com/rsadiq/salary">Salary Dataset</a>

## Algorithm Implementation

The K-means++ clustering algorithm is implemented using the `KMeans` class from scikit-learn. The algorithm is trained and tested on the Salary dataset. The following steps are performed:

1. Connect to Google Drive (optional): The notebook connects to Google Drive to access the dataset file.

2. Import the required libraries: The necessary libraries, including scikit-learn, pandas, numpy, and matplotlib, are imported.

3. Visualize the data: The Salary dataset is visualized using scatter plots to understand the data distribution.

4. Calculate the Within-Cluster Sum of Squares (WCSS) and Silhouette Score: The algorithm is run for different values of K (number of clusters), and the WCSS and Silhouette Score are calculated to determine the optimal number of clusters.

5. Plotting the Elbow Method and Silhouette Score: The WCSS scores are plotted against the number of clusters to find the "elbow" point, which indicates the optimal number of clusters. The Silhouette Score is also plotted to validate the cluster quality.

6. Train the model: The K-means++ algorithm is trained on the Salary dataset with the chosen number of clusters.

7. Prediction: The algorithm predicts the cluster labels for each data point.

8. Visualization of Clusters: The clusters are visualized using scatter plots, with each cluster assigned a different color.

9. Normalization: The dataset is normalized using Min-Max scaling to bring all features within a specific range.

10. Re-training the model: The K-means++ algorithm is re-trained on the normalized dataset.

11. Visualization of Clusters after Normalization: The clusters are visualized again after normalization to observe any changes in the clustering pattern.

12. Cluster Centroids: The coordinates of the cluster centroids are displayed.

13. Sum of Square Error (SSE): The SSE is calculated for different values of K and plotted to assess the quality of the clustering.

14. Model Persistence: The trained K-means++ model is saved for future use.

## Conclusion

The project demonstrates the implementation of the K-means++ clustering algorithm for unsupervised learning using Python. By analyzing the WCSS, Silhouette Score, and SSE, the optimal number of clusters can be determined. The algorithm helps in understanding the underlying patterns and structures in the dataset, which can be useful in various applications such as customer segmentation, anomaly detection, and recommendation systems.

Please note that the provided code is for educational purposes and may need modifications or additional steps depending on your specific use case or dataset.


