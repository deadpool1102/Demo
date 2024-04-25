install.packages("ltm")
library(ltm)

FactorAnalysis <- read.csv(file.choose(), header = TRUE)
str(FactorAnalysis)

FactorAnalysis=FactorAnalysis[9:22]
str(FactorAnalysis)

# Checking missing values
colSums(is.na(FactorAnalysis))

# Remove rows with missing values
FactorAnalysis <- na.omit(FactorAnalysis)

cronbach.alpha(FactorAnalysis)

cronbach.alpha(FactorAnalysis,CI=TRUE)


# Load the data set
#Check for the missing value
colSums(is.na(FactorAnalysis))


#Correlation 
m=cor(FactorAnalysis)
install.packages("corrplot")
library(corrplot)
corrplot(m,method="circle")

install.packages("psych")
library(psych)
# Kmo value is high inf=dicates the factor anlysis is good 
KMO(FactorAnalysis)

cortest.bartlett(FactorAnalysis)

#Using eigen values we construsct scree plot

v=eigen(m)
plot(v$values,type="b")
# No of factor =5
??factnal
#Model
factanal(FactorAnalysis,5)
f=factanal(FactorAnalysis,5,rotation = "varimax")

#Gives maximum weightage to loadings
f=factanal(FactorAnalysis,5,rotation = "promax")

#Diagramatic representation of factors
loads<-f$loadings
fa.diagram(loads)

#Factor score
out<-factanal(x=FactorAnalysis,factors = 5,scores = "regression")
out$scores
