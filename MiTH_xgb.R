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
#we have many variables such as district code and region code to represent 
#geographic location ,remove column Village- #7,remove regionname -#5

#Waterpoint_type and Waterpoint_type_group are similar variables since 
#Waterpoint_type_group has less levels 
#remove Waterpoint_type variable-#5
prop.table(table(given_data$Waterpoint_type))
prop.table(table(given_data$Waterpoint_type_group))

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


given_data_final$Statuslabel <- ifelse(given_data_final$Status == "functional", "g", "h")
given_data_final$Statuslabel<-as.factor(as.character(given_data_final$Statuslabel))


# Divide the data into train and validation
set.seed(123)

train_RowIDs = createDataPartition(given_data_final$Status,p=0.7,list=F)
train = given_data_final[train_RowIDs,]
validation= given_data_final[-train_RowIDs,]
test=test_data_final

rm(given_data_final,given_data_imputed,given_data_mod,given_data_cat,given_data_num)
rm(test_data_final,test_data_imputed,test_data_mod,test_data_cat,test_data_num)
rm(cat_Attr,num_Attr)

#Build an ensemble model with xgboost
library(xgboost)

str(train)
train_matrix <- xgb.DMatrix(data = as.matrix(train[, !(names(train) %in% c("Status", "Statuslabel"))]), 
                            label = as.matrix(train[, names(train) %in% "Status"]))

validation_matrix <- xgb.DMatrix(data = as.matrix(validation[, !(names(validation) %in% c("Status", "Statuslabel"))]), 
                                 label = as.matrix(validation[, names(validation) %in% "Status"]))

xgb_model_basic <- xgboost(data = train_matrix, max.depth = 2, eta = 1, nthread = 2, nround = 500, objective = "binary:logistic", verbose = 1, early_stopping_rounds = 10)

xgb.save(xgb_model_basic, "xgb_model_basic")

rm(xgb_model_basic)

xgb_model_basic <- xgb.load("xgb_model_basic")


basic_preds <- predict(xgb_model_basic, validation_matrix)

#Choosing the cut off
basic_preds_labels <- ifelse(basic_preds < 0.5, 0, 1)

library(caret)
result<-confusionMatrix(basic_preds_labels, validation$Status)

F1<-result$byClass[7]

params_list <- list("objective" = "binary:logitraw",
                    "eta" = 0.1,
                    "early_stopping_rounds" = 10,
                    "max_depth" = 6,
                    "gamma" = 0.5,
                    "colsample_bytree" = 0.6,
                    "subsample" = 0.65,
                    "eval_metric" = "logloss",
                    "silent" = 1)

xgb_model_with_params <- xgboost(data = train_matrix, params = params_list, nrounds = 500, early_stopping_rounds = 20)

basic_params_preds <- predict(xgb_model_with_params, validation_matrix)

basic_params_preds_labels <- ifelse(basic_params_preds < 0.5, 0, 1)

result_bf_tuning<-confusionMatrix(basic_params_preds_labels, validation$Status)

F1_bf_tuning<-result_bf_tuning$byClass[7]

#Variable Importance

variable_importance_matrix <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model_with_params)

xgb.plot.importance(variable_importance_matrix)

sampling_strategy <- trainControl(method = "repeatedcv", number = 5, repeats = 2, verboseIter = F, allowParallel = T)

param_grid <- expand.grid(.nrounds = 40, .max_depth = c(2, 4, 6), .eta = c(0.1, 0.3),
                          .gamma = c(0.6, 0.5, 0.3), .colsample_bytree = c(0.6, 0.4),
                          .min_child_weight = 1, .subsample = c(0.5, 0.6, 0.9))

xgb_tuned_model <- train(x = train[ , !(names(train) %in% c("Statuslabel", "Status"))], 
                         y = train[ , names(train) %in% c("targetlabel")], 
                         method = "xgbTree",
                         trControl = sampling_strategy,
                         tuneGrid = param_grid)

xgb_tuned_model$bestTune

plot(xgb_tuned_model)

tuned_params_preds <- predict(xgb_tuned_model, validation[ , !(names(validation) %in% c("Statuslabel", "Status"))])

result_Validation<-confusionMatrix(tuned_params_preds, validation$Statuslabel)

F1_Validation<-result_Validation$byClass[7]

tuned_params_preds_Test <- predict(xgb_tuned_model, test[ , !(names(test) %in% c("Statuslabel", "Status"))])

#Using the model built, predict the values for test data

basic_params_preds_labels <- ifelse(tuned_params_preds_Test == "g", 0, 1)

#Write to the output file

output<-data.frame(test_data$Id,basic_params_preds_labels)

summary(output)

colnames(output)<-c("Id","Status")

write.csv(output,file="Samplesubmission3.csv",row.names = F)

