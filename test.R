anomaly_detector <- function(feature,Z=3){
    results<-c()
    diff<-c()
    n <- 0
    i=0
    X = sort(feature)
    for (j in feature) {
      if (i==0){
        value = j
      }
      else{
        value = feature[i] - feature[i-1]
      }
      diff<-append(diff,value)
      i = i + 1
    }
    X_mean = mean(diff)
    X_std = sd(diff)
    X_upper = X_mean + X_std*Z
    X_lower = X_mean - X_std*Z
    i = 0
    for (j in diff){
      if(X_upper < j || X_lower > j){
        results<-append(results,"Outlier")
        n = n +1
      }
    else{
      results<-append(results,"Standard")}
      i = i+1
    }
    plot(as.ts(feature),ylim=range(-3:3)) #random
    abline(h=X_upper, col="#e15f3f", lwd=2)
    abline(h=X_lower, col="#e15f3f", lwd=2)

    
    position = data.frame(id=seq(1, length(feature)), value=feature)
    anomalyH = position[position$value > X_upper, ]
    anomalyH = anomalyH[!is.na(anomalyH$value), ]
    anomalyL = position[position$value < X_lower, ]
    #anomalyL = anomalyL[!is.na(anomalyL$value)]
    anomaly = data.frame(id=c(anomalyH$id, anomalyL$id),
                         value=c(anomalyH$value, anomalyL$value))
    points(x = anomaly$id, y =anomaly$value, col="#e15f3f")
    return(results)
    }
    