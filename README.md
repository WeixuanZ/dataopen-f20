# Gentrification Prediction and Analysis 

_Submission to 2020 European Regional Data Open._

The data included in this repository is tracked using Git LFS, please ensure it is installed before cloning. All the dependencies can be installed through

```bash
pip install -r requirements.txt && yarn install
```

GPU acceleration can be enabled by running

```bash
plaidml-setup
```

The LSTM network is trained on data from 2009 to 2017, when predicting values for 2018 on a testing dataset (25% split), it has acuracy of 0.95.

A Random Forest regressor is used to predict home value based on other features from the same year of the same tract, fitted on data from 2009 to 2018. It is then used to predict home values using the predictions by the nn, the mean squared error between which and home values predicted by the nn is 0.00267. The correlation between the two is 0.865.

The predicted data can be found in `data/census_predict.csv`.

Interactive map available at https://weixuanz.github.io/dataopen2020/.


---

This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau.

To requests new data, export your api key as an environemnt variable `CENSUS_API_KEY`.

List of 2009 ACS5 variables https://api.census.gov/data/2009/acs/acs5/variables.html
(warning: if you're using Chrome, you'll need 1.3 GB of RAM just to load this page)
If you don't mind JSON, then use the file in the data directory.
