#loading the dataset and reading the file
heart_data = read.csv(file = "/Users/jyoticaa/Desktop/SA heart_030.csv")


#exploring the data
dim(heart_data)
#we have 11 columns and 462 rows in the dataset 
names(heart_data)
#chd is the target variable here in our dataset

head(heart_data)
tail(heart_data)



#checking the issing values in the dataset
sum(is.na(heart_data))
#the dataset says we have zero missing values in the dataset


#scaling the variables
heart_data = subset(heart_data,select = -c(row.names_030))
dim(heart_data)

heart_data$famhist_030 = ifelse(heart_data$famhist_030 == 'Present',1,0)
str(heart_data)

scale_chd = scale(subset(heart_data,select = -c(chd_030)))
head(scale_chd)

chd1 = cbind(scale_chd,heart_data$chd_030)
chd1



#dividing the dataset into training and testing set  
library(caTools)
set.seed(190) 
sample = sample.split(heart_data$chd_030, SplitRatio = .75)
train = subset(heart_data, sample == TRUE)
test  = subset(heart_data, sample == FALSE)



#verifying the splitting of our data
prop.table(table(train$chd_030))
prop.table(table(test$chd_030))
#from the result we can say that the data has been distributed fairly in our training and testing set

#1
#Logistic regression is useful when we are predicting a binary outcome from a set of continuous predictor varibles 
modelfit1 = glm(chd_030 ~ sbp_030 + tobacco_030 + ldl_030 + adiposity_030 + famhist_030 + typea_030 + obesity_030 + alcohol_030 + age_030,
                data = train,family = binomial)
summary(modelfit1)

#if the value of p is < 0.05 it refers that those variables are significant with respect to our target variable
#Since the Z wald test for tobacco_010 ,ldl_010,famhist_010,typea_010 and age_010 are lesser than level of significance 0.05
# Therefore these features are  significant in model building.

#Hence, we will fit the model only with the significant factors only
#2
modelfit2 = glm(chd_030 ~  tobacco_030 + ldl_030 + famhist_030 + typea_030 + age_030,
                data = train,family = binomial)
summary(modelfit2)

#3
#final_model = -7.74111 +  0.06196 (tobacco_030) +  0.19360 (ldl_030)  + 1.05475(famhist_030)
               # +  0.04544(typea_030) +  0.06426(age_030)

# A lower AIC score indicates superior goodness-of-fit and a lesser tendency to over-fit.
#in the second model we can see that we have achieved a lower AIC value 
#in the first model we got the AIC value 361.27 and in the second model we got the AIC value 353.7
#which is not very low but a considerate amount since we eliminated all other variables whose p value was greater than 0.05.

#4
#fitting and predicting the first model
pred = predict(modelfit1,newdata= test)
library(caret)
pred = ifelse(pred>0.5,1,0)
confusionMatrix(factor(pred),factor(test$chd_030))
#here the correct predicted values are 71 and 15; these observations does not have the disease and have the disease respectively
#25 observations were predicted that they dont have chd but actually they did had chd 
#5 people were predicted that they do have chd but in reality they did not had it 
#the 25 observations worries us the most here
#the accuracy of the model is also only 74.14% which is quite low 


#we calculate the precision of the confusion matrix which is Also called Positive predictive value
#The ratio of correct positive predictions to the total predicted positives.
p1 = table(factor(pred),factor(test$chd_030))
precision(p1)
# the precision of the above confusion matrix is 73.95%

#fitting and predicting the second model
pred2 = predict(modelfit2,newdata= test)
pred2 = ifelse(pred2>0.5,1,0)
confusionMatrix(factor(pred2),factor(test$chd_030))
#here we can see that 72 observations were predicted who did not have the disease and they were correctly predicted
#14 observations were predicted to have the disease and they did had the disease
#26 observations were predicted that they did not had the disease but in reality they DID had the disease
#4 observations were predicted to have the disease but in reality they did not had the disease
#here, according to me what worries us the most is the 26 observations because these people will be under the impression 
#that they are not suffering from the disease but they are
#Also, we cannot take risk with the lives of 26 people 
#the model accuracy is 74.14% which according to me is very low with respect to the domain which we have as our dataset


#we calculate the precision of the confusion matrix which is Also called Positive predictive value
#The ratio of correct positive predictions to the total predicted positives.
p2 = table(factor(pred2),factor(test$chd_030))
precision(p2)
#we got the precision value to be 73.46% 



#to conclude here i can say that the precision of both the models is almost same and the accuracy of both the
#models is very low keeping the domain of our dataset in mind 

#b (1)
library(rpart)
cart = rpart(chd_030~.,data = train,method = "class")
print(cart)


library(rpart.plot)
rpart.plot(cart,main= "Classification Tree")
# here age ,famhist,typea,ldl,tobacco and sbp are the features which are helping in classifing the chd as 0 or 1.
#the accuracy here is 68.97%
p3 = predict(cart,newdata = test,type = 'class')
p3
tab = table(predicted = p3,actual = test[,10])
tab
confusionMatrix(tab)
precision(tab)
#the precision of the cart is 74.39% and the accuracy of the model is only 68% 



#b(2)
library(randomForest)

rf = randomForest(factor(chd_030)~.,data = train)
summary(rf)


varImpPlot(rf) #this is the variable importance plot  
getTree(rf)
p4 = predict(rf,newdata = test,type = 'response')
p4
tab2 = table(p4,test[,10])
confusionMatrix(tab2)
precision(tab2)
#the precision for random forest is 70.11%
#which is still low than the first and the second model
#the accuracy is 64.66%

#b(3)
#the best classifier among 1 and 2 would be CART as the accuracy of CART is greater than than of 
#Randomforest
#but the difference between the accuracies is only almost 4% which is not a major difference


#b(4)
#the precision of the model CART is 74.39% which is also greater than the random forest model 
#but since precision shows the accuracy of the positive class, with respect to our domain of the dataset
#the precision value 74.39% is very low

#age will be the splitting criteria from the diagram which we have got





