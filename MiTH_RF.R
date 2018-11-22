#Clear the environment 

rm(list=ls(all=TRUE))

library(tidyverse) # data manipulation
library(mlr)       # ML package (also some data manipulation)
library(knitr)     # just using this for kable() to make pretty tables

#Read the input data that is given

setwd("D:/INSOFE/MiTH")

# Load required libraries
library(caret)
library(rpart)

#Read the files
train_data<-read.csv("train.csv",na.strings = c(""," ","?","-","NA","<NA>",NA))

train_labels<-read.csv("trainlabels.csv",header=T)

colnames(train_labels)<-c("Id","Status")

given_data <- merge(train_data,train_labels,by="Id")

test_data<-read.csv("test.csv",na.strings = c(""," ","?","-","NA","<NA>",NA))
#Use head() and tail() functions to get a feel of the data

head(given_data)

tail(given_data)

head(test_data)

tail(test_data)

#Check the structure of the input data

str(given_data)

str(test_data)



#Check the distribution of the input data using the summary function

summary(given_data)

summary(test_data)

colSums(is.na(given_data))

colSums(is.na(test_data))

#Check the proportion of functional/non-functional pipes
prop.table(table(given_data$Status))

prop.table(table(given_data$Waterpoint_type))


#Barplot of functional and non-functional pipes
barplot(table(given_data$Status))

barplot(table(given_data$Waterpoint_type_group))

barplot(table(given_data$Source_type))

boxplot(Gps_height~Status, data = given_data, xlab ="Status", ylab = "Gps_height", main = "Gps_height v/s Status")


#ggplot

ggplot(given_data, aes(x = Region_code, y = Gps_height, color = Status)) +
  geom_point()


#Remove the columns id-#1,target-#35 .Also removed the 
#Organization_surveyed -#15 column since the data is same for all rows
#Waterpointname has too many levels and looks like it is just a name given 
#to the waterpoint

#Village is categorical variable and has many levels since 
#we have many variables such as district code and region code to represent 
#geographic location ,remove column Village- #7

#Waterpoint_type and Waterpoint_type_group are similar variables since 
#Waterpoint_type_group has less levels 
#remove Waterpoint_type variable-#5
prop.table(table(given_data$Waterpoint_type))
prop.table(table(given_data$Waterpoint_type_group))

#Remove variable Region Code- #8,Ward_name-#10 and Organization_funding - #14 and SchemeName -#17
#and Company_installed - #19. Since these are just names and it wont contribute to 
#the prediction.



#Management and Management_group are similar variables since 
#Management_group has less levels 
#remove Management variable-#20
prop.table(table(given_data$Management))
prop.table(table(given_data$Management_group))

#Extraction_type and Extraction_type_group are similar variables since 
#Extraction_type_group has less levels 
#remove Extraction_type variable-#22
prop.table(table(given_data$Extraction_type))
prop.table(table(given_data$Extraction_type_group))


#Quantity and Quantity_group are similar and has same no. of levels
#So remove Quantity_group -#30
prop.table(table(given_data$Quantity))
prop.table(table(given_data$Quantity_group))

#Source_type and Source are similar variables since Source_type has less levels 
#So remove Source variable-#31
prop.table(table(given_data$Source_type))
prop.table(table(given_data$Source))


#Water_quality and Quality_group are similar variables in which Quality_group 
#has some levels combined(salty,salty abandoned as salty) and (fluoride,fluoride abandoned as fluoride)
#So Water_quality is removed -#27
prop.table(table(given_data$Water_quality))
prop.table(table(given_data$Quality_group))

given_data_mod<-given_data[,-c(1,4,5,7,8,10,14,15,17,19,20,22,27,30,31,35)]

test_data_mod<-test_data[,-c(1,4,5,7,8,10,14,15,17,19,20,22,27,30,31)]


 
#Imputing the variables Public_meeting and Permit to have only two levels
given_data_mod$Public_meeting[is.na(given_data_mod$Public_meeting)]<-FALSE
given_data_mod$Permit[is.na(given_data_mod$Permit)]<-FALSE

test_data_mod$Public_meeting[is.na(test_data_mod$Public_meeting)]<-FALSE
test_data_mod$Permit[is.na(test_data_mod$Permit)]<-FALSE




#Changing the Public_meeting and Permit to categorical variable

given_data_mod$Public_meeting<-as.factor(given_data_mod$Public_meeting)
given_data_mod$Permit<-as.factor(given_data_mod$Permit)

test_data_mod$Public_meeting<-as.factor(test_data_mod$Public_meeting)
test_data_mod$Permit<-as.factor(test_data_mod$Permit)


prop.table(table(given_data_mod$Region_code))
prop.table(table(given_data_mod$District_code))

# Splitting into categorical and numerical attributes

num_Attr<-c("Amount_of_water","Gps_height","Population")

cat_Attr<-setdiff(x = colnames(given_data_mod), y = num_Attr)

given_data_cat <- subset(given_data_mod,select =cat_Attr)

given_data_mod[,cat_Attr] <- data.frame(apply(given_data_cat, 2, function(x) as.factor(as.character(x))))

given_data_cat<-given_data_mod[,cat_Attr] 

given_data_num<-given_data_mod[,num_Attr] 

test_data_cat <- subset(test_data_mod,select =cat_Attr)

test_data_mod[,cat_Attr] <- data.frame(apply(test_data_cat, 2, function(x) as.factor(as.character(x))))

test_data_cat<-test_data_mod[,cat_Attr] 

test_data_num<-test_data_mod[,num_Attr] 

given_data_mod<-cbind(given_data_cat,given_data_num)

test_data_mod<-cbind(test_data_cat,test_data_num)
#Imputation

library(DMwR)

given_data_imputed <- centralImputation(data = given_data_mod)

given_data_final<-cbind(given_data_imputed,Status=given_data$Status)

sum(is.na(given_data_final))

test_data_imputed<-centralImputation(data = test_data_mod)

test_data_final<-test_data_imputed

sum(is.na(test_data_final))

#Changing the Public_meeting and Permit to categorical variable

given_data_final$Public_meeting<-as.factor(given_data_final$Public_meeting)
given_data_final$Permit<-as.factor(given_data_final$Permit)

test_data_final$Public_meeting<-as.factor(test_data_final$Public_meeting)
test_data_final$Permit<-as.factor(test_data_final$Permit)



# Divide the data into train and validation

set.seed(123)

train_RowIDs = createDataPartition(given_data_final$Status,p=0.7,list=F)
train = given_data_final[train_RowIDs,]
validation= given_data_final[-train_RowIDs,]
test=test_data_final



rm(given_data_final,given_data_imputed,given_data_mod,given_data_cat,given_data_num)
rm(test_data_final,test_data_imputed,test_data_mod,test_data_cat,test_data_num)



#########3Random Forest

#Install Packages
library(randomForest)
# Model Building -

#set.seed(123)

# Build the classification model using randomForest
rf_model = randomForest(Status ~ ., data=train, 
                        keep.forest=TRUE, ntree=100) 

# Print and understand the model
print(rf_model)

# Important attributes
rf_model$importance  
round(importance(rf_model), 2) 


# Extract and store important variables obtained from the random forest model
rf_Imp_Attr = data.frame(rf_model$importance)
rf_Imp_Attr = data.frame(row.names(rf_Imp_Attr),rf_Imp_Attr[,1])
colnames(rf_Imp_Attr) = c('Attributes', 'Importance')
rf_Imp_Attr = rf_Imp_Attr[order(rf_Imp_Attr$Importance, decreasing = TRUE),]

# plot (directly prints the important attributes) 
varImpPlot(rf_model)


#Predict on Train data 
pred_Train = predict(rf_model, 
                     train[,setdiff(names(train), "Status")],
                     type="response", 
                     norm.votes=TRUE)

# Build confusion matrix and find accuracy   
cm_Train = table("actual"= train$Status, "predicted" = pred_Train);
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
rm(pred_Train, cm_Train)

# Predicton Validation Data
pred_Validation = predict(rf_model, validation[,setdiff(names(validation),
                                                        "Status")],
                          type="response", 
                          norm.votes=TRUE)

# Build confusion matrix and find accuracy   
cm_Validation = table("actual"=validation$Status, "predicted"=pred_Validation);
accu_Validation= sum(diag(cm_Validation))/sum(cm_Validation)
rm(pred_Validation, cm_Validation)

accu_Train
accu_Validation
rf_Imp_Attr$Attributes

# Build randorm forest using all attributes. 
top_Imp_Attr = as.character(rf_Imp_Attr$Attributes[1:13])

set.seed(15)

# Build the classification model using randomForest
model_Imp = randomForest(Status~.,
                         data=train[,c(top_Imp_Attr,"Status")], 
                         keep.forest=TRUE,ntree=100) 

# Print and understand the model
print(model_Imp)

# Important attributes
model_Imp$importance  

# Predict on Train data 
pred_Train = predict(model_Imp, train[,top_Imp_Attr],
                     type="response", norm.votes=TRUE)


# Predicton Test Data
pred_Validation= predict(model_Imp, validation[,top_Imp_Attr],
                         type="response", norm.votes=TRUE)
table(pred_Validation)

library(caret)

result_Train<-confusionMatrix(pred_Train, train$Status)

result_Train

result_Train$byClass[7]

result_Validation<-confusionMatrix(pred_Validation, validation$Status)

result_Validation

result_Validation$byClass[7]

#Select mtry value with minimum out of bag(OOB) error.

str(train)
mtry <- tuneRF(train[,-20],train$Status, ntreeTry=100,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

set.seed(71)
rf <- randomForest(Status~.,data=train, mtry=best.m, importance=TRUE,ntree=100)
print(rf)

#Evaluate variable importance
importance(rf)

# Important attributes
rf$importance  
round(importance(rf), 2)   

# Extract and store important variables obtained from the random forest model
rf_Imp_Attr = data.frame(rf$importance)
rf_Imp_Attr = data.frame(row.names(rf_Imp_Attr),rf_Imp_Attr[,1])
colnames(rf_Imp_Attr) = c('Attributes', 'Importance')
rf_Imp_Attr = rf_Imp_Attr[order(rf_Imp_Attr$Importance, decreasing = TRUE),]

# Predict on Train data
# Predict on Train data 

pred_Train = predict(rf, 
                     train[,-20],
                     type="response", 
                     norm.votes=TRUE)


# Predicton Test Data

pred_Validation = predict(rf, validation[,-20],
                          type="response", 
                          norm.votes=TRUE)


library(caret)

result_Train<-confusionMatrix(pred_Train, train$Status)

result_Train

result_Train$byClass[7]

result_Validation<-confusionMatrix(pred_Validation, validation$Status)

result_Validation

levels(test$Waterpoint_type)<-levels(train$Waterpoint_type)
levels(test$Region_code)<-levels(train$Region_code)
levels(test$Scheme_management)<-levels(train$Scheme_management)
levels(test$Waterpoint_type_group)<-levels(train$Waterpoint_type_group)

#Using the model built, predict the values for test data
pred_Test  =  predict(rf_model, test[,-20],
                      type="response", 
                      norm.votes=TRUE)

#Write to the output file

output<-data.frame(test_data$Id,pred_Test)

summary(output)

colnames(output)<-c("Id","Status")

write.csv(output,file="Samplesubmission5.csv",row.names = F)

