
library(tidyverse)

datDir = '/Users/liuq13/SPCD_data'
all_odor_pred_df = NULL

for(j in 1:42){
  
  SubID = paste0('Sub',j,'_done')
  outdir = file.path(datDir,SubID,'beh')
  
  
  
  odor_pred_file = file.path(outdir,'odor_pred_dat.RData')
  
  if (!file.exists(odor_pred_file)) { 
    next
  } 
  
  load(odor_pred_file) 
  
  avg_odor_pred_df = tmp_df %>% 
    group_by(Run) %>%
    reframe(mean_acc = mean(acc,na.rm=T)) %>%
    mutate(SubID = j)
    
  all_odor_pred_df = rbind(all_odor_pred_df,avg_odor_pred_df)
  
}


ggplot(all_odor_pred_df,mapping = aes(x=Run,y=mean_acc, label = SubID)) +
  geom_jitter(height = 0,width = 0.1, color = "blue") +
  geom_text(vjust = -0.5, size = 3) +
  labs(x = "Run", y = "Mean Accuracy", title = "Mean Accuracy per Run")





