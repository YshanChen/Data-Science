# ==============================
#           smbinning
# ==============================

# Package loading and data exploration
library(smbinning) 
data(chileancredit) 
str(chileancredit) 
table(chileancredit$fgood) # Tabulate target variable

# Training and testing samples
train_index = sample(1:nrow(chileancredit),nrow(chileancredit)*0.7)
chileancredit.train = chileancredit[train_index,]
chileancredit.test = chileancredit[-train_index,]

# [smbinning]
result=smbinning(df=chileancredit.train,y="fgood",x="cbs1",p=0.05) # Run and save result
result$ivtable # Tabulation and Information Value
result$iv # Information value
result$cuts 
# [1] 558 581 622 646 704
summary(chileancredit$cbs1)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
#   400.0   605.0   643.0   642.3   680.0   850.0     986 
result$bands # Bins or bands 
# [1] 440 558 581 622 646 704 850 (区间)
result$ctree # Decision tree from partykit

# [smbinning.gen] 
# Generate new binned characteristic into a existing data frame
chileancredit.train=
  smbinning.gen(chileancredit.train,result,"g-cbs1") # Update training sample
class(chileancredit.train$`g-cbs1`) #"factor"
levels(chileancredit.train$`g-cbs1`)
# [1] "00 Miss"   "01 <= 558" "02 <= 581" "03 <= 622" "04 <= 646" "05 <= 704" "06 > 704"
chileancredit=
  smbinning.gen(chileancredit,result,"g-cbs1") # Update population

# [smbinning.factor]
chileancredit.train$inc
result.train=smbinning.factor(df=chileancredit.train,y="fgood",x="inc")
# [1] "Too many categories"

chileancredit.train$pmt
result.train=smbinning.factor(df=chileancredit.train,y="fgood",x="pmt")
result.train$ivtable
result.test=smbinning.factor(df=chileancredit.test,y="fgood",x="pmt")
result.test$ivtable

# [smbinning.plot]
par(mfrow=c(2,2))
smbinning.plot(result.train,option="dist",sub="pmt Level (Tranining Sample)")
smbinning.plot(result.train,option="badrate",sub="pmt Level (Tranining Sample)")
smbinning.plot(result.test,option="dist",sub="pmt Level (Test Sample)")
smbinning.plot(result.test,option="badrate",sub="pmt Level (Test Sample)")

