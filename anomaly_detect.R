install.packages("Rcpp")
install.packages("ecp")
install.packages("devtools")
install.packages(c('devtools','curl'))

devtools::install_github("twitter/BreakoutDetection")

library(Rcpp)
library(ecp)
library(BreakoutDetection)

createDay = function(noise=0) {
  point=seq(0, pi, by=0.02)
  connection=sin(point)
  noise = rnorm(length(point), sd = noise)
  return(connection+noise)
}

createDays = function(totalDays, noise=0) {
  allDays = c()
  for (day in 1:totalDays ) {
    allDays = c(allDays, createDay(noise))
  }
  return(allDays)
}

set.seed(1234)
p1 = createDays(3, 0.2)
anomaly = createDays(1, 0)*2 + rnorm(158, sd = 0.07)
days = c(p1, anomaly)
plot(as.ts(days))

# EDM - fail
res = breakout(days, min.size=158, method='multi', beta=.001, degree=1, plot=TRUE)
res$plot

# E-Divisive - fail
ediv = e.divisive(as.matrix(days), min.size=158, alpha=1)
plot(as.ts(days))
abline(v=ediv$estimates,col="blue")