set.seed(4)
data <- read.csv("/home/bharath/full_datasets/Completed/fact_price/fact_price_weekly_smp.csv",stringsAsFactors=FALSE)
str(data)

date <- strptime(data$date,format="%Y%m%d")
data$curr_value <-as.numeric(data$curr_value)

price <- data$curr_value
new_data <- data.frame(date,price)

new_data$price[is.na(new_data$price)] = mean(new_data$price, na.rm = T)

plot(as.ts(new_data$price))

#install.packages("FBN")
library(FBN)

decomposed_days = decompose(ts(new_data$price, frequency = 7), "multiplicative")
plot(decomposed_days)


random = decomposed_days$random
min = mean(random, na.rm = T) - 4*sd(random, na.rm = T)
max = mean(random, na.rm = T) + 4*sd(random, na.rm = T)

plot(as.ts(as.vector(random)), ylim = c(-0.5,2.5))
abline(h=max, col="#e15f3f", lwd=2)
abline(h=min, col="#e15f3f", lwd=2)


#find anomaly
position = data.frame(id=seq(1, length(random)), value=random)
anomalyH = position[position$value > max, ]
anomalyH = anomalyH[!is.na(anomalyH$value), ]
anomalyL = position[position$value < min, ]
anomalyL = anomalyL[!is.na(anomalyL$value), ]
anomaly = data.frame(id=c(anomalyH$id, anomalyL$id),
                     value=c(anomalyH$value, anomalyL$value))
anomaly = anomaly[!is.na(anomaly$value), ]

plot(as.ts(new_data$price))
real = data.frame(id=seq(1, length(new_data$price)), value=new_data$price)
realAnomaly = real[anomaly$id, ]
points(x = realAnomaly$id, y =realAnomaly$value, col="#e15f3f")

library(stats)
trend = runmed(new_data$price, 7)
plot(as.ts(trend))

detrend = new_data$price / as.vector(trend)
plot(as.ts(detrend))
m = t(matrix(data = detrend, nrow = 7))
seasonal = colMeans(m, na.rm = T)
random = new_data$price / (trend * seasonal)
rm_random = runmed(random[!is.na(random)], 3)

min = mean(rm_random, na.rm = T) - 5*sd(rm_random, na.rm = T)
max = mean(rm_random, na.rm = T) + 5*sd(rm_random, na.rm = T)
plot(as.ts(random)) #random
abline(h=max, col="#e15f3f", lwd=2)
abline(h=min, col="#e15f3f", lwd=2)


position = data.frame(id=seq(1, length(random)), value=random)
anomalyH = position[position$value > max, ]
anomalyH = anomalyH[!is.na(anomalyH$value), ]
anomalyL = position[position$value < min, ]
#anomalyL = anomalyL[!is.na(anomalyL$value)]
anomaly = data.frame(id=c(anomalyH$id, anomalyL$id),
                     value=c(anomalyH$value, anomalyL$value))
points(x = anomaly$id, y =anomaly$value, col="#e15f3f")
results = anomaly_detector(random)


