#Clear the environment 

rm(list=ls(all=TRUE))

library(tidyverse) # data manipulation
library(mlr)       # ML package (also some data manipulation)
library(knitr)     # just using this for kable() to make pretty tables
library(xgboost)
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
#we have many variables such as Regionname,district code and region code to represent 
#geographic location ,remove column Village- #7

#Waterpoint_type and Waterpoint_type_group are similar variables since 
#Waterpoint_type_group has less levels 
#remove Waterpoint_type variable-#8
table(given_data$Waterpoint_type)
table(given_data$Waterpoint_type_group)

#Remove variable Ward_name-#10 and Organization_funding - #14 and SchemeName -#17
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

given_data_mod<-given_data[,-c(1,4,7,8,10,14,15,17,19,20,22,27,30,31,35)]

test_data_mod<-test_data[,-c(1,4,7,8,10,14,15,17,19,20,22,27,30,31)]

#Changing the Public_meeting and Permit to categorical variable

given_data_mod$Public_meeting<-as.factor(as.character(given_data_mod$Public_meeting))
given_data_mod$Permit<-as.factor(as.character(given_data_mod$Permit))

test_data_mod$Public_meeting<-as.factor(as.character(test_data_mod$Public_meeting))
test_data_mod$Permit<-as.factor(as.character(test_data_mod$Permit))

#Imputing the variables Public_meeting and Permit to have only two levels
given_data_mod$Public_meeting[is.na(given_data_mod$Public_meeting)]<-FALSE
given_data_mod$Permit[is.na(given_data_mod$Permit)]<-FALSE

test_data_mod$Public_meeting[is.na(test_data_mod$Public_meeting)]<-FALSE
test_data_mod$Permit[is.na(test_data_mod$Permit)]<-FALSE
prop.table(table(given_data_mod$Region_code))
prop.table(table(given_data_mod$District_code))

#given_data_mod$Gps_height[(given_data_mod$Gps_height<0)] <-0

#prop.table(table(given_data_mod$Basin_name))

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

#sum(is.na(test_data_final))


# Divide the data into train and validation
set.seed(123)

train_RowIDs = createDataPartition(given_data_final$Status,p=0.7,list=F)
train = given_data_final[train_RowIDs,]
validation= given_data_final[-train_RowIDs,]
test=test_data_final

rm(given_data_final,given_data_imputed,given_data_mod,given_data_cat,given_data_num)
rm(test_data_final,test_data_imputed,test_data_mod,test_data_cat,test_data_num)
rm(cat_Attr,num_Attr)

#----------------C50-------------------- 

library(C50)

# Build C5.0 model on the training dataset

c50_tree<-C5.0(Status ~ ., train)

c50_Model = C5.0(Status~.,train,rules=T)

C5imp(c50_tree,metric="usage")
summary(c50_Model)
plot(c50_Model)


# Using C5.0 Model predicting with the train dataset
c50_Train = predict(c50_Model, train, type = "class")
c50_Train = as.vector(c50_Train)
table(c50_Train)

result_Train<-confusionMatrix(c50_Train, train$Status)
result_Train
result_Train$byClass[7]

# Using C50 Model prediction on validation dataset 
c50_Validation = predict(c50_Model, validation, type = "class")
c50_Validation = as.vector(c50_Validation)

result_Validation<-confusionMatrix(c50_Validation, validation$Status)

result_Validation
result_Validation$byClass[7]

cm_C50 = table(c50_Validation, validation$Status)
sum(diag(cm_C50))/sum(cm_C50)

#Predict for the test data 

c50_Test = predict(c50_Model, test, type = "class")
c50_Test = as.vector(c50_Test)

#Write to the output file

output<-data.frame(test_data$Id,c50_Test)

summary(output)

colnames(output)<-c("Id","Status")

write.csv(output,file="Samplesubmission1.csv",row.names = F)


