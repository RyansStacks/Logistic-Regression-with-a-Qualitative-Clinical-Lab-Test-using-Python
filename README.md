# Logistic Regression with a Qualitative Clinical Lab Test using Python
### (January 2026)  [Python]

#### by Ryan Breen, M.S. MLS (ASCM)


![image.png](image.png)


## Background
A Spectrophotometer measures light absorption/transmission to quantify substances using units of measure called Optical Density (OD). Optical Density measures how much light a material blocks, calculated as the logarithm (base 10) of the ratio of incident light to transmitted light (OD = -log(T)), with higher values meaning less light passes through


A specific case of Spectrophotometer analysis is called "ELISA (Enzyme-Linked Immunosorbent Assay)" and is a plate-based technique using antibodies and enzymes to detect and measure specific proteins (like antigens or antibodies) in samples, often relying on a specialized spectrophotometer (an ELISA reader) to read the color changes in microplate wells, linking light intensity to substance concentration. 


The Optical Density units increase or decrease based on a specific concenctration(i.e. ng/mL) of the measurand (measured substance) and this direct relationship may be capture using approaches such as linear regression to create an equation to convert input OD units to measurement units such as ie. ng/mL. 

However, often is the case that ELISA test are used to determine if a testing sample is simply positive or negative for the sample present. In such cases, the quantitative method of using linear regression that outputs continuous values (i.e. 100) does not provide a helpful functionality. In turn, logistic regression is used as a way to incorporate regressional analysis to determine the outcome but outputs values between 0 to 1 representing the predicted probability that an outcome is positive. 

The outputs values between 0 to 1 are literally created by inputting values into a regression equation but transforming the regression equation into a __logit function__ so continuous values ranging from -∞ to +∞ are converted to values between 0 to 1. For example, a value of 0.1 would indicate that the output has a predicted probability of 10% of being positive and classically a cutoff of 50% or 0.5 would be used to determine if such a value was negative if below such threshold. Furthermore, the __logit function__ creates outputs that are sigmoidal or S-shaped so that outputs are drastically classified as negative or positive based on predicated probabilities.




#### Logistic Regression Equation:

$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n)}}$ 

![image.png](image.png)

#### Receiver Operator Curves

To determine an ELISA cutoff using logistic regression, you model the probability of a sample being positive (based on its optical density, or OD) against a known outcome (positive/negative status), then use the resulting model (often via ROC curve analysis) to find the OD value that best balances sensitivity and specificity, or meets clinical goals, rather than relying on a fixed 0.5 probability threshold. This involves plotting Receiver Operating Characteristic (ROC) curves, choosing a threshold that maximizes the Area Under the Curve (AUC) or balances Type I/II errors, and then identifying the corresponding OD value as the cutoff. 

A Receiver Operating Characteristic (ROC) curve is a graph that visualizes a binary classification model's performance across all possible thresholds, plotting its True Positive Rate (Sensitivity) (TPR) against its False Positive Rate (FPR) at various settings. 

![image.png](image.png)

The graph below demonstrates a __Receiver Operating Curve__ that is commonly displayed when performing logistic regression equation.  The __Receiver Operating Curve__ using the __Area Under the Curve (AUC)__ as a performance measure to determine how well the logistic regression equation is classifying positive and negative results using the inputs and outputs. A perfect classifier would have an AUC = 1 meaning that the __True Positive Rate__ = 100% (1.00) and __False Positive Rate__ = 0% (0.00). Most statistical software are capable of measuring the AUC:

![image.png](image.png)

#### Peformance Characteristics Equations for Logistic Regression Classification
The following equations gauge specific characteristics of the regression model. These measures may only be used if we know the true classificaiton of each sample. For example, if we are preforming an HIV ELISA test and the patient truely has been diagnosed with HIV and the test is positive then the result is a _True Positive_. 

![image.png](image.png)

## Peforming Logistic Regression with Python

#### Install Libraries
Python `Sci-Kit Learn` and `Pandas` are used to create dynamic models that can upload `.csv` files or `Excel Files` to create report metrics and graphs in seconds.


```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
```

#### Create Mock Data
The table below displays outcomes as 0 or 1 for negative or positive results based on respective OD values


```python

# --- Example data ---
df = pd.DataFrame({
    "outcome": [0,1,0,1,1,0,1],
    "OD": [0.12, 0.85, 0.30, 1.10, 0.95, 0.22, 1.25]
})

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>outcome</th>
      <th>OD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1.10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1.25</td>
    </tr>
  </tbody>
</table>
</div>



#### Create Logistic Regression Model
The logistic regression model is created by _fitting_ the X (OD values) and y (outcomes) inside a `Python` `SciKit Learn` function `LogisticRegression()`
Below, by inputing OD values into the equation, one is able to get a predicated probability that the OD value is positive.


```python
# --- Fit logistic regression ---
X = df[["OD"]]
y = df["outcome"]
model = LogisticRegression()
model.fit(X, y)

coefficients = model.coef_[0]
intercept = model.intercept_[0]


print("The logistic regression equation is:")
print(f"Predicted Probability = 1 / 1 + exp(-({round(intercept,3)} + {round(coefficients[0],3)} *   OD)))")

```

    The logistic regression equation is:
    Predicted Probability = 1 / 1 + exp(-(-0.438 + 1.082 *   OD)))
    

#### Creating the ROC Curve and Determining the OD Cutoff value from the Area Under the Curve (AUC)
An ROC Curve is created from the logistical regression equation we find above by using `Scikit Learn` `roc_curve` function that finds the threshold or point at which the logistic regression equation best classifies negative and positive results with the highest __Youde score__. The __Youden score__ = tpr - (1 - fpr) and demonstrates the ability of the model to distinguish true positives from false positives.


```python
df["prob"] = model.predict_proba(X)[:,1]

fpr, tpr, thresholds = roc_curve(y, df["prob"])
youden = tpr - (1 - fpr)
best_idx = np.argmax(youden)

prob_cutoff = thresholds[best_idx]
print("The cutoff is:", round(prob_cutoff, 2))
```

    The cutoff is: 0.42
    

#### Receiver Operator Curve


```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

roc_auc = auc(fpr, tpr)

# Plot
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")  # diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```


    
![png](output_24_0.png)
    


#### Area Under the Curve (AUC)


```python
from sklearn import metrics
roc_auc = metrics.auc(fpr, tpr)
print("The AUC is:", round(roc_auc, 2))

```

    The AUC is: 1.0
    

The cutoff value has an absorbance value of 0.42 meaning that are ELISA test with an OD value greater than or equal to 0.42 would be considered positive for the condition being tested.

#### Classical Approach: Determining the Level of Blank for Qualitative Assays

Obtain OD Readings: Measure the OD readings for a set of known negative control samples (representing samples without the target antigen) and positive control samples (representing samples with the target antigen). Additionally, measure the OD readings for the samples being tested.

Calculate Mean and Standard Deviation: Calculate the mean and standard deviation of the OD readings for the negative control samples. This represents the background signal of the assay.

Determine Cutoff Value: The cutoff value is often calculated as a multiple (e.g., 2 or 3) of the standard deviation above the mean OD of the negative control samples. Alternatively, it can be set at a specific OD value determined by the assay's characteristics or previous validation studies.

Cutoff Value = Mean OD of negative controls + (X * Standard Deviation of negative controls)

X represents the number of standard deviations above the mean (e.g., X = 2 or 3 for a cutoff at 2 or 3 standard deviations above the mean).

Interpret Results: Samples with OD readings above the cutoff value are considered positive, indicating the presence of the target antigen. Samples with OD readings below the cutoff value are considered negative.

#### Classical Approach (3SD Above Negative Sample Mean) vs Logistical Regression Cutoff (Using AUC)


```python
cutoff_classical = np.mean(df[df['outcome'] == 0]['OD']) + 3 * np.std(df[df['outcome'] == 0]['OD'])

print("The classical cutoff is: ", round(cutoff_classical,2))
print("The logistical regression cutoff is: ", round(prob_cutoff,2))
```

    The classical cutoff is:  0.43
    The logistical regression cutoff is:  0.42
    

## Conclusion
There are two major approaches that may be utilized to determine the cutoff value for an ELISA assay. Logistic Regression provides a robust method to determine the value and may even incorporate covariates such as gender, smokers, and other demographical information that is typically analyzed with the clinical assay. The other approach is more simplistic but provides a much faster, easier, and cost-effective approach as it only involves testing negative samples only where the 2 or 3 standard deviations is added to the mean of such samples. This guide should serve an overview for how a manufacturer or laboratory may determine cutoff values for qualitative ELISA testing.


