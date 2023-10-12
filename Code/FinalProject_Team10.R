#Loading libraries
library(rpart,quietly = TRUE)
library(caret,quietly = TRUE)
library(rpart.plot,quietly = TRUE)
library(rattle)
library(lares)
library(kableExtra)
library(tidyverse)
library(randomForest)
library(mlbench)
library(caTools)
library(lares)
library(heatmaply)
library(leaflet.extras)

# Loading the dataset
#ref: https://stackoverflow.com/questions/32556967/understanding-r-is-na-and-blank-cells
USaccidents <- read.csv("Datasets/US_Accidents_Dec21_updated.csv", na.strings = c("", "NA"))
#View(USaccidents)

# Knowing the data set
dim(USaccidents)
str(USaccidents)

# Overview of the data set
tail(USaccidents, 6)
summary(USaccidents)

# checking for duplicate
dim(USaccidents[duplicated(USaccidents$id),])[1]

# Checking for missing and NA values
nrow(na.omit((USaccidents)))
nrow(USaccidents) - sum(complete.cases(USaccidents))

#new england USaccidents
states_NewEngland <- c("RI", "MA", "VT", "ME", "CT", "NH")
NE_USaccidents <- USaccidents %>% filter(State %in% states_NewEngland)
rm(USaccidents)
dim(NE_USaccidents)
#str(NE_USaccidents)


# Finding the unique values in the columns
unique_count <- sapply(lapply(NE_USaccidents, unique), length, simplify=FALSE)
singleValueCols <- data.frame(unique_count[!(unique_count > 1)])
colnames(singleValueCols)

# Removing columns with factor = 1
NE_USaccidents <- select(NE_USaccidents, -all_of(colnames(singleValueCols)))
dim(NE_USaccidents)

#Checking for missing data(NA) percentage in each column
missing_NA <- round((colMeans(is.na(NE_USaccidents)))*100,2)
order <- order(missing_NA,decreasing = TRUE)
missing_NA_ordered <- missing_NA[order]
head(missing_NA_ordered,3)

#Dropping columns with high NA values
drop_Cols <- c("Number", "Precipitation.in.", "Wind_Chill.F." )
NE_USaccidents <- select(NE_USaccidents, -all_of(drop_Cols))
NE_USaccidents <- na.omit(NE_USaccidents)
dim(NE_USaccidents)


#checking correlation btw all numeric variables and Severity
NE_USaccidents_numCols <- NE_USaccidents %>% select_if(is.numeric)
cor_numCols <- cor(NE_USaccidents_numCols[ ,colnames(NE_USaccidents_numCols) != "Severity"],
                NE_USaccidents_numCols$Severity)
cor_numCols 

#checking correlation btw all boolean variables and Severity
#https://stackoverflow.com/questions/22772279/converting-multiple-columns-from-character-to-numeric-format-in-r
colnames(NE_USaccidents)
NE_USaccidents_facCols <- select(NE_USaccidents,
                                 c(26:42)) %>% mutate_if(is.character,as.factor)
colnames(NE_USaccidents_facCols)
#str(NE_USaccidents_facCols)
NE_USaccidents_FaNum <- NE_USaccidents_facCols %>% mutate_if(is.factor, as.numeric)
#str(NE_USaccidents_FaNum)
NE_USaccidents_numCols2 <- cbind(NE_USaccidents$Severity, NE_USaccidents_FaNum)
#str(NE_USaccidents_numCols2)
colnames(NE_USaccidents_numCols2)[1] <- 'Severity'

x1 <- cor(NE_USaccidents_numCols2[ ,colnames(NE_USaccidents_numCols2) != "Severity"],
          NE_USaccidents_numCols2$Severity)
cor_numCols2 <- as.data.frame(x1)
colnames(cor_numCols2) <- c("r")

cor_numCols2 <- tibble::rownames_to_column(cor_numCols2, "Variables")
arrange(cor_numCols2,r) %>% 
  kable(caption = "Correlation of Severity with other Variables",
        align = "lc",
        table.attr = "style='width:50%;'") %>%
  kable_classic_2(font_size = 14)


# Correlation between independent variables
corr_cross(NE_USaccidents,
           max_pvalue = 0.05,
           top = 15 )

# Listing the columns to be dropped based on multicolinearity results
drop_Cols_2 <- c("ID","End_Lat","End_Lng","Start_Lng",
                 "Civil_Twilight","Nautical_Twilight",
                 "Astronomical_Twilight","Start_Lng",
                 "City","State","County","Wind_Direction",
                 "Weather_Timestamp","Description","Street",
                 "Traffic_Calming","Airport_Code",
                 "Weather_Condition","Crossing",
                 "Start_Time","End_Time","Timezone","Zipcode")

# removing the columns from the dataset
NE_USaccidents_1 <- select(NE_USaccidents,-all_of(drop_Cols_2) )
dim(NE_USaccidents_1) #44048 obs. 20 var /43972


#Data Exploration

# Visualizing Severity
barplot(table(NE_USaccidents_1$Severity),
        main = "Severity Analysis of Accidents in New",
        xlab = "Severity",
        ylab = "Count of Incidents",
        col = "lightblue")


#Temperature, Wind Speed and Humidity distribution for severity level
par(mfrow = c(1, 3))
plot(NE_USaccidents_1$Severity, NE_USaccidents_1$Wind_Speed.mph., xlab= "Accident Severity", ylab = "Wind Speed in Miles per hour")
plot(NE_USaccidents_1$Severity, NE_USaccidents_1$Humidity..., xlab= "Accident Severity", ylab = "Humidity in percentage")
plot(NE_USaccidents_1$Severity, NE_USaccidents_1$Temperature.F., xlab= "Accident Severity", ylab = "Temperature in Fahrenheit")



#PREPARING THE DATA

# Converting Severity into Low and High
NE_USaccidents_1$Severity<- as.factor(ifelse(NE_USaccidents_1$Severity<=2,"Low","High"))

#data splicing
set.seed(12345)
train <- sample(1:nrow(NE_USaccidents_1),
                size = ceiling(0.80*nrow(NE_USaccidents_1)),
                replace = FALSE)

# Creating the training set
NE_USaccidents_train <- NE_USaccidents_1[train,]
dim(NE_USaccidents_train) # 35239 obs. 20 var /35178    20
table(NE_USaccidents_train$Severity)

# Creating the test set
NE_USaccidents_test <- NE_USaccidents_1[-train,]
dim(NE_USaccidents_test) # 8809 obs.  20 var /8794   20
table(NE_USaccidents_test$Severity)


# MODEL - 1
# Classification Tree

# Finding the perfect split
number.perfect.splits <- apply(X=NE_USaccidents_1[-1],
                               MARGIN = 2,
                               FUN = function(col){
                                 t <- table(NE_USaccidents_1$Severity,col)
                                 sum(t == 0)})

# Descending order of perfect splits
order <- order(number.perfect.splits,decreasing = TRUE)
number.perfect.splits <- number.perfect.splits[order]

# Building the classification tree with rpart
rpart_tree <- rpart(Severity~.,
              data = NE_USaccidents_train,
              method = "class")

# Visualize the decision tree with rpart.plot
rpart.plot(rpart_tree, nn=TRUE)

# choosing the best complexity parameter "cp" to prune the tree
cp.optim <- rpart_tree$cptable[which.min(rpart_tree$cptable[,"xerror"]),"CP"]

# tree prunning using the best complexity parameter
rpart_tree_optim <- prune(rpart_tree, cp=cp.optim)

# Visualize the decision tree with rpart.plot
rpart.plot(rpart_tree_optim, nn=TRUE)


#Testing the model
rpart_tree_pred <- predict(object = rpart_tree_optim, 
                           NE_USaccidents_test[-1],
                           type="class")


#Calculating accuracy
acc_rpart_pred <- table(NE_USaccidents_test$Severity, rpart_tree_pred) 
confusionMatrix(acc_rpart_pred) 


# MODEL - 2
#Random Forest model building

#Splitting the dataset
Y_train <- NE_USaccidents_train[,'Severity']
Y_test <- NE_USaccidents_test[,'Severity']
X_train <- NE_USaccidents_train[, !(colnames(NE_USaccidents_train) == 'Severity')]
X_test <- NE_USaccidents_test[, !(colnames(NE_USaccidents_test) == 'Severity')]


#Random Forest modelling for Severity of Accident
set.seed(123)
ne_accidents_rf = randomForest(x = X_train,
                               y = Y_train,
                               mtry=12,
                               importance=TRUE,
                               ntree = 500)


# Predicting the Test set results
y_pred_ne = predict(ne_accidents_rf, newdata = NE_USaccidents_test)
plot(y_pred_ne, NE_USaccidents_test$Severity)
abline(0,1)

#Confusion Matrix
confusionMatrix(y_pred_ne, NE_USaccidents_test$Severity)

#Calculating the test mean value
mean(y_pred_ne== NE_USaccidents_test$Severity)

#Finding the Variable of Importance
importance(ne_accidents_rf)
varImp(ne_accidents_rf)

#Plot MeanDecreaseAccuracy and MeanDecreaseGini
varImpPlot(ne_accidents_rf)



# MODEL - 3
#Gradient boosting model building

#gbm model 1
#Referred from rpubs.com: https://rpubs.com/billantem/651903
GBM_1<- train(Severity~Start_Lat+Distance.mi.+Temperature.F.+ Pressure.in.+Visibility.mi.+Station+
                Sunrise_Sunset + Amenity+Bump+Junction+No_Exit+Humidity...+Side+Give_Way+Roundabout+
                Railway+Traffic_Signal+Wind_Speed.mph.,
              data = NE_USaccidents_train,
              method = "gbm",
              trControl = trainControl(method="CV", repeats = 10),
              preProcess = c("center", "scale"))

predicted_1 <- predict(GBM_1,
                       newdata = NE_USaccidents_test)
confusionMatrix(predicted_1, NE_USaccidents_test$Severity)


summary(GBM_1, las=1)

#gbm model 2
#Referred from rpubs.com: https://rpubs.com/billantem/651903

GBM_2 <- train(Severity~Start_Lat+Distance.mi.+Temperature.F.+ Station+
                 Sunrise_Sunset + Amenity + Junction +Humidity...+Side+Give_Way+
                 Traffic_Signal+Wind_Speed.mph.,
               data = NE_USaccidents_train,
               method = "gbm",
               trControl = trainControl(method="CV", repeats = 10),
               preProcess = c("center", "scale"))

predicted_2 <- predict(GBM_2,
                       newdata = NE_USaccidents_test)
confusionMatrix(predicted_2, NE_USaccidents_test$Severity)

summary(GBM_2, las=1)



#Visualizations to support our analysis

##Visualizing the accident hotspot locations
severity_high <- NE_USaccidents_1 %>% filter(NE_USaccidents_1$Severity=="high")
map <- severity_high %>%
  leaflet () %>%
  addTiles () %>%
  addHeatmap(lng=~Start_Lng,
             lat=~Start_Lat,
             intensity=2,
             blur=4,
             max=1,
             radius=4)
map


#Distance distribution for severity level
summary(ne_accidents_fs$Distance.mi.)
plot(NE_USaccidents_1$Distance.mi.~NE_USaccidents_1$Severity,
     xlab= "Accident Severity",
     ylab = "Distance in Miles")


#Barplot for severity levels in various states (4 levels)
ggplot(NE_USaccidents, aes(fill=Severity, y=Temperature.F., x=State)) +
  geom_bar(position='dodge', stat='identity')
