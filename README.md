# Dimensionality-Reductioin-algorithms-Implimentation-of-PCA-LDA-ICA-SVD
In this we are going to learn about how to Dimensionality-Reductioin-algorithms-PCA-LDA-ICA-SVD
# DIMENTIONALITY REDUCTION
- Many machine learning problems have thousands or even millions of features for each training instance. Not only does this make training extremely slow, it can also make it much harder to find a good solution.
- Reducing dimensionality does lose some information (just like compressing an image to JPEG can degrade its quality), so even though it will speed up training, it may also make your system perform slightly worse.
- For example, in face recognition, the size of a training image patch is usually larger than 60 x 60 , which corresponds to a vector with more than 3600 dimensions.
- In some cases , reducing the dimensionality of the training data may filter out some noise and unnecessary details and thus result in higher performance (but in general it won’t; it will just speed up training).
## Practical reasons
- Redundancy reduction and intrinsic structure discovery
- Intrinsic structure discovery
- Removal of irrelevant and noisy features
- Feature extraction
- Visualization purpose
- Computation and Machine learning perspective
## PCA (Principle component analysis)
- Pca is by far the most popular dimensionality algorithm which is in use.<br>
- The main idea of principal component analysis (PCA) is to reduce the dimensionality of a data set consisting of many variables correlated with each other, either heavily or lightly, while retaining the variation present in the dataset, up to the maximum extent. The same is done by transforming the variables to a new set of variables, which are known as the principal components (or simply, the PCs).
- PCA is basically the Linear Algebra.
- By simple example will help to understand what is it and how is it works.
- If we take 100*2 matrix
- Then we have two choice 1.is to standardized this data and 2.without standardized.
<p>So 1st we are going to understand with standardization,</p>
<p>Now for implementing PCA</p>

- Step1:
```python
# generating a random data of 100*2
height = np.round(np.random.normal(1.75, 0.20, 10), 2)
weight = np.round(np.random.normal(60.32, 15, 10), 2)
Data = np.column_stack((height, weight))
print("printing the Data:")
```

- Step2:
<p>Now find mean of this Data column wise</p>

```python
Mean =np.mean(Data,axis=0)
print("Mean of this Data:" + str(Mean))
```

- Step3:
<p>Now find standard variation of Data</p>

```python
Std = np.std(Data, axis=0)
print("Standard Deviation of this Data:" + str(Std))
```

- Step4:
<p>Now Standardized this data and find Co-variance matrix</p>

```python
stdData = (Data - Mean) / Std
print("Our Stdandized matrix is :" + str(stdData))
print(stdData.shape)
```

<p>find Co-variance matrix</p>

```python
covData = np.cov(stdData.T)
print("Our Co-variance matrix is:" + str(covData))
```

- Step5:
<p>find eighen values and eighen vectors</p>

```python
values, vectors = eig(covData)
print(values)
print(vectors)
```

- Step6:
```python 
i=0
pairs=[(np.abs(values[i]), vectors[:,i]) for i in range(len(values))]
print(pairs)
print("------------------------------------------------------------------------------------------------------------------")
pairs.sort(key=lambda x: x[0], reverse = True)
print(pairs)
``` 
<p>above we have pair the eighen values with eighen vectors and sorted them by eighen values</p>
<p>Now in this When we have taken mnist-dataset and applied above line of code then we are getting -nan error during standardization and it's because in mnist dataset 60000*784 the some variables are colinear with each other .</p>

# LDA (Linear Discriminant Analysis)














## PCA vs LDA
<p>Now as we have seen two methods let's compare both of them on various datasets like wine,digits and iris datasets and visualize the plot of the results.</p>
<p>Hear we are going to use sklearn library's datasets and decomposition function for PCA and LDA.</p>
- Importing dataset

```python 
#for iris
iris = datasets.load_iris()
print (iris)
#for wine Dataset
X = wine.data
y = wine.target
target_names = wine.target_names
#for digits dataset
X = digits.data
y = digits.target
target_names = digits.target_names
```
- Calculating PCA and LDA with the help of sklearn library function

```python
#for PCA
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
print(X_r)
#for LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)
print(X_r2)
```
<p>hear n_components cannot be larger than min(n_features, n_classes - 1).hear i have given example about iris dataset Using min(n_features, n_classes - 1) = min(4, 3 - 1) = 2 components.</p>
<p>now for differantiate between this two i have made plots of both the results from that you can see the difference</p>
- lda_vs_pca plot of thoese three databases


    
