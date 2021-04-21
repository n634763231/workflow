raw_data <- read.csv("data_raw/iris_raw.csv", check.names = FALSE) # 读取原始数据
clean_data <- subset(iris,Species%in%c("versicolor","virginica"))#choose versicolor and virginica
write.csv(clean_data, "data_clean/iris_clean.csv")