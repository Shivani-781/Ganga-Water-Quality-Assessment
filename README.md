# Ganga-Water-Quality-Assessment-Using-Dempster-Shafer-Theory
## Overview
This project is based on my work as a Research Intern under the supervision of <b>Dr. Manish Kumar</b>, Associate Professor at IIIT, Allahabad.

In this work, we will study the water samples from <b>the holy river of Ganga</b> which flows in the city of <b>Prayagraj</b>, situated at the confluence of the Rivers Ganga, Yamuna and Saraswati. The determination of water quality has been an important environmental problem. This study proposes an idea to develop a <b>machine learning model</b> which can accurately assess the quality of river water using <b>data mining techniques</b>.


## Data About Data
The parameters of the time-series dataset collected from the river Ganga using daily measure of sensors include: <b>Electrical Conductivity, Dissolved Oxygen (DO), Oxidation Reduction Potential (ORP), pH and Temperature of water</b>.

Let's learn more about each of these physicochemical parameters:
<ul>
  <li>The <b>conductivity</b> of water is measured by its capability to pass electrical flow. Conductivity is quite an essential parameter to assess water quality as it is dependent on the concentration of conductive ions in the water.</li>
  <li>The amount of <b>oxygen dissolved</b> in the water can tell a lot about its quality. As the dissolved oxygen (DO) increases, the temperature of water decreases. It is usually measured in both milligrams per litre and percentage saturation.</li>
  <li>Another quality parameter is <b>Oxidation Reduction Potential</b>, which is also known as <b>REDOX</b>. It is a measurement that reflects the ability of a molecule to oxidise or reduce another molecule.</li>
  <li>The relative amount of free hydrogen and hydroxyl ions determines the <b>pH of water</b>. In other words, it is a measure of how acidic or basic is the given water sample. pH ranges from 0-14, with 7 being neutral.</li>
  <li><b>Temperature</b> is yet another important factor to consider when assessing water quality. Temperature can alter the physical and chemical properties of water.</li>
</ul>

## Problem Statement
Complete survey surmised that the water quality is measured from spatial-temporal changes of water quality, statistical analysis of quality parameter and satellite data of the Ganga River basin.

Also, the traditional water quality monitoring approaches give precise measurements, but their costs are expensive and they are time-consuming. Moreover, these methods did not complete regional needs. Also, it requires large travelling and laboratory expenses, especially for a large area.

So, it is very challenging to <b>analyse water quality in real-time</b>. For solving these problems, we are trying to make a water quality assessment model based on the intrinsic nature of the collected <b>IoT data</b>.

## Methodology
### DATA PRE-PROCESSING: (Transformation, Cleaning, Extraction, Normalization)
The <b>transformations</b> performed on the data using NumPy and Pandas libraries include:
<ul>
  <li>Changing the data types of Date and Time attributes</li>
  <li>Dropping unnecessary attributes</li>
  <li>Sorting the time-series data based on the Date attribute</li>
  <li>Transforming the resulting data using pivot tables so that each sensor becomes an attribute</li>
</ul>  

The sensing data collected includes data of Ganga as well as Sangam river. Based on the sensing details, the details of Ganga river have been extracted and the negative values are replaced by NaN values, to handle the <b>missing data</b>. The missing data in each attribute is then replaced by the mean of the last five values of the particular attribute using the <b>forward fill</b> technique.

The details of Ganga river have been extracted into <b>samples</b> for further processing. Negative values are dropped using the python libraries. As per the report on sensing details, the data is extracted into 21 buckets using the Pandas library.

In order to understand the underlying behaviour of our attributes, we need to apply the <b>Gaussian distribution formula</b> to check whether the data is normally distributed. On plotting the probability distribution if a <b>bell-shaped curve</b> is formed and the mean, mode, and median are equal then the variable is normally distributed. Normal distribution is dependent on two parameters which are mean and the standard deviation. If the data exhibits normal distribution, it is feasible to be forecasted with higher accuracy. We find that the collected data for oxidation-reduction potential of Ganga river water does not follow normal distribution as its mean is greater than 1.

We apply the <b>Central Limit Theorem</b>, which states that as the sample size is increased, the sampling distribution of the mean reaches a normal distribution. The random samples from the extracted data, each of sample size greater than equal to 30, are then checked for normal distribution using the Gaussian distribution formula. We find that the bell-shaped curve for oxidation-reduction potential data has a high mean while others have mean less than 1.

We normalize the data as variables that are measured at different scales do not contribute equally to the model fitting. Thus, to deal with this potential problem feature-wise normalization is usually used before model fitting. So, the data is normalized using <b>MinMaxScaler()</b> from scikit-learn library.

### DATA CLUSTERING: (K-Means Clustering)
Before fitting the model, we need to classify the data using <b>K-Means Clustering</b>, which is an extensively used technique for data clustering.

The K-means algorithm begins with a first group of randomly selected centroids. This cluster is used as the beginning points for every cluster. Then iterative calculations are done to optimize the positions of the centroids. It stops when there is no change in the values of centroids. 

The <b>elbow method</b> computes an average score for all clusters. It runs k-means clustering for a range of values for k (say from 1-10).

From the plot for conductivity and dissolved oxygen attributes, we see a few points of one cluster overlaps with the points of another cluster. To get better results, we apply <b>SVM Kernel Tric</b>k to classify & model the data and visualize them by plotting their <b>Receiver Operating Characteristic (ROC) Curves</b>.

### DATA MODELLING: (Dempster Shafer Theory)
The data is modelled using the <b>Dempster Shafer Theory</b> with the help of machine learning models like <b>SVM, Logistic Regression, Naïve Bayes and Decision Tree</b>.

Now, let's have a look on all of these models.

<b>Support Vector Machine (SVM)</b> is a supervised machine learning algorithm which can be used for both classification and regression problems. A line or a hyperplane in a high or infinite dimensional space is constructed. A <b>Kernel Trick</b> is a simple method where a non-linear data is projected onto a higher dimension space so as to make it easier to classify the data where it could be linearly divided by a plane.

<b>Logistic regression</b> is a supervised learning classification algorithm used to predict the probability of a target variable. Based on the number of categories, it can be classified as: Binomial, Multinomial and Ordinal. Logistic regression is a powerful algorithm that utilizes a sigmoid function or a logistic function.

<b>Naive Bayes Classifier</b> is a supervised machine learning algorithm that uses the Bayes’ Theorem, which relies on the naive assumption that input variables are independent of each other. <b>Gaussian Naive Bayes</b> supports continuous data. When working with continuous data, an assumption often taken is that the continuous values associated with each class are distributed according to a normal (or Gaussian) distribution.

<b>Decision tree</b> is a type of supervised learning algorithm that is mostly used in classification problems. A decision tree makes decisions by splitting nodes into sub-nodes. This process is performed multiple times during the training process until only homogenous nodes are left.

The <b>Dempster Shafer Theory</b>, also known as the theory of <b>belief functions</b>, was designed to mathematically model and validate the uncertainty involved in statistical inferences. It combines a set of representations and model data when there is a lack of information.

DST is mostly known to represent <b>uncertainties or imprecision in a hypothesis</b>. The hypotheses characterize all the possible states of the system. These hypotheses are assigned a <b>probability mass assignment (PMA)</b> which when combined leads to a decision.

The process of forming mass assignment function and combining the same is thus crucial for <b>accurate prediction</b>. The high-level features are converted into Dempster-Shafer mass functions by aggregating them using Dempster’s rule of combination.

## Conclusion
Among the classifiers, <b>Decision Tree</b> obtained highest accuracy, which indicates that the model can accurately assess the quality of water samples of River Ganga.

In future, we can use extended dataset and some other machine learning technique to assess the quality of water.

Also, with <b>extended dataset</b>, we would be able to extract the pattern of the river water at a particular interval of time over a year by using predictive analysis. The forecasting of river flow has great importance in water resources.
