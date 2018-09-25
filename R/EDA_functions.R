library(ggplot2)
library(gridExtra)
library(dplyr)
library(magrittr)

"""
Todo:
1. 增加多变量关系的数据探索功能
"""

"""
近针对单变量X和Y进行分析
1. X=连续值，Y=连续值 => 对X箱型图、密度图
2. X=离散值，Y=连续值 => 对X和Y箱型图、密度图
3. X=连续值，Y=离散值 => 对X和Y箱型图、密度图
4. X=离散值，Y=离散值 => 对X和Y直方图、气泡图（个数与占比）
"""

# EDA - 近针对单变量
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
    density_plot = ggplot(data, aes(x=data[[target]], fill=data[[x]])) + 
      geom_density(position="identity", alpha=0.2) +
      xlab(target) + scale_fill_discrete(name=x)
    
    boxplot = ggplot(data, aes(x=data[[x]], y=data[[target]])) + geom_boxplot() +
      xlab(x) + ylab(target)
    
    grid.arrange(density_plot, boxplot, ncol=2)
    # print(boxplot)
  }
  
  # X=连续值，Y=离散值 => 对X和Y箱型图、密度图
  if (is.numeric(data[[x]]) & is.factor(data[[target]])){
    density_plot = ggplot(data, aes(x=data[[x]], fill=data[[target]])) + 
      geom_density(position="identity", alpha=0.2) +
      xlab(x) + theme(legend.position = "none")
    
    boxplot = ggplot(data, aes(x='x', y=data[[x]], fill=data[[target]])) + geom_boxplot() + 
      xlab('') + ylab(x) + scale_fill_discrete(name=target)
    
    grid.arrange(density_plot, boxplot, ncol=2)
    # print(boxplot)
  }
  
  # X=离散值，Y=离散值 => 对X和Y直方图、气泡图（个数与占比）
  if (is.factor(data[[x]]) & is.factor(data[[target]])){
    data_cnt = data[,c(x, target)]
    colnames(data_cnt) = c('x', 'target')
    data_cnt = data_cnt %>% select(x, target) %>% group_by(target, x) %>% summarise(count=n())
    
    # 气泡图
    pop = ggplot(data_cnt, aes(x=x, y=target, size=count), alpha=0.2) + 
      geom_point() + xlab(x) + ylab(target)

    # 直方图
    data_cnt_per = data_cnt
    t1 = data_cnt_per %>%  group_by(target) %>% summarise(cnt_gpby_target=sum(count))
    t2 = data_cnt_per %>% group_by(x) %>% summarise(cnt_gpby_x=sum(count))
    data_cnt_per = merge(x = data_cnt_per, y = t1, by = 'target', all.x=TRUE)
    data_cnt_per = merge(x = data_cnt_per, y = t2, by = 'x', all.x=TRUE)
    data_cnt_per = data_cnt_per %>% mutate(per_gpby_target = count/cnt_gpby_target,
                                           per_gpby_x = count/cnt_gpby_x)
    
    his_gpby_x = ggplot(data_cnt_per, aes(x=x, y=per_gpby_x, fill=target)) + 
      geom_bar(stat='identity', position="fill") + 
      xlab(x) + ylab(paste('per_gpby_', x, sep=''))
    his_gpby_target = ggplot(data_cnt_per, aes(x=target, y=per_gpby_target, fill=x)) + 
      geom_bar(stat='identity', position="fill") + 
      xlab(target) + ylab(paste('per_gpby_', target, sep='')) + 
      scale_fill_discrete(name=x)
    
    lay = rbind(c(1,1,1,3,3), c(2,2,2,3,3))
    grid.arrange(pop, his_gpby_x, his_gpby_target, layout_matrix = lay)

  }
  
  
}

# Test -----------------------------------
data = iris

# 创建一个离散值
data$Sepal.Length_bin = factor(round(data$Sepal.Length,0))
levels(data$Sepal.Length_bin)

eda(data = data, x='Sepal.Width', target = 'Sepal.Length')
eda(data = data, x='Sepal.Length_bin', target = 'Sepal.Length')
eda(data = data, x='Sepal.Length', target = 'Species')
eda(data = data, x='Sepal.Length_bin', target = 'Species')
















