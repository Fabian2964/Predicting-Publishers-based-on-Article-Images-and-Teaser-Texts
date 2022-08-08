# Project Write-up – Predicting Newspaper Publishers based on Article Images and Teaser Texts

## Abstract
The goal of this project was to use neural networks to predict newspaper publishers for article images and article teaser texts published online to more deliberately control images and introductory texts of published stories to draw audience interests. Additionally, the model may provide evidence for journalists which article images and teaser text characteristics are used by competitors. My analysis was based on images and teaser texts of 4 German publishers (F.A.Z., Spiegel Online, Zeit Online and SZ Online) published in the category “politics” on their main homepage Feb’22 to Jul’22. In feature engineering, I converted images and teasers to numerical arrays. In terms of modeling, features were fed to convolutional neural net, LSTM and several pre-trained models by using transfer learning. Finally, models were integrated into a web app on streamlit enabling to pass images and teaser texts to predict publishers.

## Design
The project topic supports a question of data-driven journalism. Feature data was scraped from publishers’ homepages and from corresponding URL of each image. After conversion to numerical arrays, baseline non-deep-learning models and neural net models were constructed and their performance compared. Lastly, correctly and falsely classified images and texts of were inspected within and between publishers. Inspecting differences in images and texts may support journalists to deliberately choose images and texts characteristics. On a more global level it may also help to detect differences between publishers with different readerships such as right- or left-wing.

![Workflow](https://user-images.githubusercontent.com/98846184/183523308-83883abd-d2e4-4ecb-84fe-286d9589bb7c.png)

![image](https://user-images.githubusercontent.com/98846184/183307200-ed756f6b-dcbf-46a1-9eb5-124147019e75.png)

![image](https://user-images.githubusercontent.com/98846184/183307203-8c6b24c3-5c9b-4915-88ec-49611c511c5d.png)

Algorithms
### Feature Engineering
    1. Convert RGB images to common shape of 150x150x3 dimensions used as input dimension for NN models
    2. Images were randomly transformed using ImageDataGenerator to increase models’ generalization capabilities
    3. Texts were tokenized to remove special characters and keep only the most 1000 frequent words
    4. Tokenized sentences were converted to sequence format and padded to maximum length of 50
    
### Models
For image classification, a logistic regression model was trained on PCA-reduced 2 dimensional dataset forming a non-deep-learning baseline model. A standard sequential Convolutional Neural Net with 6 layers and 642,824 parameters was constructed as deep-learning baseline model. Subsequently, pre-trained models Xception, VGG16, mobilenet were imported using transfer learning and trained on publishers’ images. For text classification LSTM model was trained as the only deep-learning model.

**Image Multiclass classification - Baseline Model**
![image](https://user-images.githubusercontent.com/98846184/183307244-61eaab87-69ac-42ff-8f2c-9e8d2bb76641.png)

**Image Binary classification - Baseline Model**
![image](https://user-images.githubusercontent.com/98846184/183307265-b2a6b6aa-c820-4706-aacb-a5d022c5e800.png)

**Text Binary classification - Baseline Model**
![image](https://user-images.githubusercontent.com/98846184/183307315-8bce3857-c5f9-4b87-bce5-80cdece27509.png)

### Model Evaluation and Selection
The entire dataset was randomly split into training, validation and test data. Models were evaluated based on their generalization performance using standard classification metrics with a focus on F1-score to judge model based on its ability to predict positives correctly among all predicted positives (precision) and to correctly identify positives among all present positives (recall). For multiclass image classification, VGG16 model had best F1-score of 0.41. Based on this, VGG16 was used for binary image classification and reached an F1-score of 0.72. For binary text classification, LSTM model had an F1-score of 0.68.

**Image Multiclass classification - Evaluation & Top Performing Model**

![image](https://user-images.githubusercontent.com/98846184/183307328-30d448fc-3e49-4f3c-9eb8-27d0fe090b9c.png)

![image](https://user-images.githubusercontent.com/98846184/183307330-212af7e8-11e0-49be-a888-68576fc251aa.png)

**Image Binary classification - Evaluation & Top Performing Model**

![image](https://user-images.githubusercontent.com/98846184/183307294-cdd033bf-fabd-4d03-ab1e-86714f93a23c.png)

![image](https://user-images.githubusercontent.com/98846184/183307296-bfa3ec2c-ca10-46b6-bf5f-e568577da5bc.png)

**Text Binary classification - Evaluation & Top Performing Model**

![image](https://user-images.githubusercontent.com/98846184/183307318-33f448fc-1115-404a-8f15-0247143c0c68.png)

![image](https://user-images.githubusercontent.com/98846184/183307321-8348ee14-337c-4ea3-9a00-fe198820f272.png)

## Tools
    • BeautifulSoup for webscraping
    • NumPy, pandas and keras for data manipulation
    • NLTK for text processing
    • Keras and sklearn for modeling
    • Matplotlib, Seaborn and Bokeh for plotting
    • Streamlit for creating a web application
    
## Web App
In addition to the slides and visuals presented, the model is embedded in a dedicated [streamlit app](https://fabian2964-deep-learning-app-image-pca3lj.streamlitapp.com/) and project outputs are placed on my personal [github](https://github.com/Fabian2964/deep_learning.git).

## Classification Examples
### Image Binary classification
![image](https://user-images.githubusercontent.com/98846184/183308197-9e22991c-c8aa-4bd2-a32d-bdae7f0445bb.png)

![image](https://user-images.githubusercontent.com/98846184/183308201-622a349c-80c3-4591-a7b9-5e36c020ddf3.png)

![image](https://user-images.githubusercontent.com/98846184/183308206-52e7e3b9-78a8-4e83-af35-99d8d5dbd3e3.png)

![image](https://user-images.githubusercontent.com/98846184/183308208-6ba25955-e401-4b20-96ae-38c87f1534fc.png)

### Text Binary classification
![image](https://user-images.githubusercontent.com/98846184/183308219-b05ff9b6-0842-4504-bbf8-51d40f11577f.png)

![image](https://user-images.githubusercontent.com/98846184/183308222-88ec6925-1738-4626-ad01-29ceb956b7a3.png)

