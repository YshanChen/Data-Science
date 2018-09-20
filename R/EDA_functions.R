setwd('/Users/chenys/Documents/Comp/ccf_unicom')
library(data.table)
library(ggplot2)
library(stringr)
library(gridExtra)
library(dplyr)
library(magrittr)

data(iris)
data = iris

table(data$Species)
data = data %>% filter(Species == 'setosa' | Species == 'versicolor')
# 创建一个离散值
data$Sepal.Length_bin = factor(round(floor(data$Sepal.Length*100)/100,0))
levels(data$Sepal.Length_bin)

# X=连续值，Y=连续值 => 对X箱型图、密度图
# X=离散值，Y=连续值 => 对X和Y箱型图、密度图
# X=连续值，Y=离散值 => 对X和Y箱型图、密度图
# X=离散值，Y=离散值 => 对X和Y直方图、气泡图（个数与占比）

# EDA
eda <- function(data, x, target){
  
  # X=连续值，Y=连续值 => 对X箱型图、密度图
  if (is.numeric(data[[x]]) & is.numeric(data[[target]])){
    boxplot = ggplot(data, aes(x='x', y=data[[x]])) + geom_boxplot() +
      xlab(x) + ylab(x)
    density_plot = ggplot(data, aes(x=data[[x]])) + 
      geom_density(position="identity") +
      xlab(x)
    grid.arrange(boxplot, density_plot, ncol=2)
    # print(boxplot)
  }
  
  # X=离散值，Y=连续值 => 对X和Y箱型图、密度图
  if (is.factor(data[[x]]) & is.numeric(data[[target]])){
    density_plot = ggplot(data, aes(x=data[[target]], fill=data[[x]], alpha=0.1)) + 
      geom_density(position="identity") +
      xlab(target) + scale_fill_discrete(name=x)
    
    boxplot = ggplot(data, aes(x=data[[x]], y=data[[target]])) + geom_boxplot() +
      xlab(x) + ylab(target)
    
    grid.arrange(density_plot, boxplot, ncol=2)
    # print(boxplot)
  }
  
  # X=连续值，Y=离散值 => 对X和Y箱型图、密度图
  if (is.numeric(data[[x]]) & is.factor(data[[target]])){
    density_plot = ggplot(data, aes(x=data[[x]], fill=data[[target]], alpha=0.1)) + 
      geom_density(position="identity") +
      xlab(x) + theme(legend.position = "none")
    
    boxplot = ggplot(data, aes(x='x', y=data[[x]], fill=data[[target]])) + geom_boxplot() + 
      xlab('') + ylab(x) + scale_fill_discrete(name=target)
    
    grid.arrange(density_plot, boxplot, ncol=2)
    # print(boxplot)
  }
  
  # X=离散值，Y=离散值 => 对X和Y直方图、气泡图（个数与占比）
  
  
}
eda(data = data, x='Sepal.Width', target = 'Sepal.Length')









