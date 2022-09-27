# Boston Housing

For details, please check the report [HERE](https://github.com/yovalishere/Real-Estate/blob/main/Boston%20Housing/Boston_Report.pdf)

### Project description
The aim of this analysis is to compare different models such as **random forest**, **boosting** and other baseline
methods (ie. **linear regression with stepwise variable selection using AIC** & **regression tree**.

<img src="https://github.com/yovalishere/Real-Estate/blob/main/Boston%20Housing/optimal_tree_real_est.jpg" width="600" height="350" />
*The above image shows the structure of the optimal regression tree used for comparison in this analysis. 

### Data description
The [‘Boston’ dataset](http://lib.stat.cmu.edu/datasets/boston) is from the library MASS. It consists of 506 rows of census data of Boston in 1970. 14 variables are present, 
with ‘medv’ being the dependent variable. 

### Project findings
The random forest and boosting models perform better than baseline methods. The performance of random forest after turning 
improved but not the case for boosting. In boosting, the number of
trees is reduced from 500 to 396 after tuning. That makes sense because boosting error should drop down as the
number of trees increases, which is evidence showing that boosting is reluctant to overfit. Overall, random forest
with tuned parameters performs the best. 

<img src="https://github.com/yovalishere/Real-Estate/blob/main/Boston%20Housing/summary_tbl_boston.jpg" width="600" height="125" />
