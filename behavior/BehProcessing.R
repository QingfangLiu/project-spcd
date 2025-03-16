
# processing subjects' data 
# with cartoon version of the task

rm(list = ls())
library(tidyverse)
library(ggforce)
library(openxlsx)

datDir = '/Users/liuq13/SPCD_data'
mapping = c('1' = "B1>B3", '2' = "B2>B3", '3' = "B1>B2", 
            '4' = "A1>A3", '5' = "A2>A3", '6' = "A1>A2")

for(j in 1:40){

SubID = paste0('Sub',j,'_done')
path <- list.files(path = file.path(datDir,SubID), pattern = "Sub*", 
                      full.names = T, recursive = F)
if (length(path)==0) { # move to next if path does not exist
  next
}

print(SubID)
outdir = file.path(datDir,SubID,'beh')
if (file.exists(outdir)) { # if 'beh' folder already exists move to next
  next
} 

dir.create(outdir)

### pre-meal probe task, 3 runs, 24 trials each
probe_file_name_pattern = 'Sub.*_Day2_Probe_pre_.*Trials\\.csv'
probe_pre_file = list.files(path = path,pattern = probe_file_name_pattern)
if (length(probe_pre_file) != 1) {
  stop("probe_pre_file not unique. Execution halted.")
} 
dat = read.csv(file.path(path,probe_pre_file))
dat$Run = as.numeric(dat$Run) # coerce Run col from char to num
dat = subset(dat,Run > 0)       # select rows
if (nrow(dat) != 72) {
  stop("Error: Trial number from pre-meal probe not 72. Execution halted.")
} 

pre_meal_dat = dat %>% 
  mutate(key_resp.keys_raw = gsub("'", "", dat$key_resp.keys_raw)) %>%
  mutate_at(vars(trial, choice_class, set, choice_pair,cue_left_idx,
                 cue_right_idx,flip,odor_left,odor_right,ITIDur,ISIDur,correct,
                 key_resp.keys_raw,key_resp.rt_raw), as.numeric) %>%
  mutate(resp = key_resp.keys_raw, 
         rt = key_resp.rt_raw,
         acc = as.numeric(correct==key_resp.keys_raw)) %>%
  select(Run,trial, choice_class, set, choice_pair,cue_left_idx,
         cue_right_idx,flip,odor_left,odor_right,ITIDur,ISIDur,correct,
         resp,acc,rt) %>%
  mutate(choice_class = plyr::mapvalues(choice_class,from=names(mapping),to=mapping))

pre_meal_dat %>%
  group_by(choice_class) %>%
  reframe(mean=mean(acc,na.rm=T))


### post-meal probe task, 3 runs, 24 trials each
probe_file_name_pattern = 'Sub.*_Day2_Probe_post_.*Trials\\.csv'
probe_post_file = list.files(path = path,pattern = probe_file_name_pattern)
if (length(probe_post_file) != 1) {
  stop("probe_post_file not unique. Execution halted.")
} 

dat = read.csv(file.path(path,probe_post_file))
dat$Run = as.numeric(dat$Run) # coerce Run col from char to num
dat = subset(dat,Run > 0)       # select rows
if (nrow(dat) != 72) {
  stop("Error: Trial number from post-meal probe not 72. Execution halted.")
} 

post_meal_dat = dat %>% 
  mutate(key_resp.keys_raw = gsub("'", "", dat$key_resp.keys_raw)) %>%
  mutate_at(vars(trial, choice_class, set, choice_pair,cue_left_idx,
                 cue_right_idx,flip,odor_left,odor_right,ITIDur,ISIDur,correct,
                 key_resp.keys_raw,key_resp.rt_raw), as.numeric) %>%
  mutate(resp = key_resp.keys_raw, 
         rt = key_resp.rt_raw,
         acc = as.numeric(correct==key_resp.keys_raw)) %>%
  select(Run,trial, choice_class, set, choice_pair,cue_left_idx,
         cue_right_idx,flip,odor_left,odor_right,ITIDur,ISIDur,correct,
         resp,acc,rt) %>%
  mutate(choice_class = plyr::mapvalues(choice_class,from=names(mapping),to=mapping))

post_meal_dat %>%
  group_by(choice_class) %>%
  reframe(mean=mean(acc,na.rm=T))

### odor prediction task: 36 trials, 3 runs
pred_file_name_pattern = 'Sub.*_Day2_OdorPred_.*PredTrials\\.csv'
pred_file = list.files(path = path,pattern = pred_file_name_pattern)
if (length(pred_file) != 1) {
  stop("pred_file not unique. Execution halted.")
} 

dat = read.csv(file.path(path,pred_file))
dat$Run = as.numeric(dat$Run) # coerce Run col from char to num
dat = subset(dat,Run > 0)       # select rows
if (nrow(dat) != 108) {
  stop("Trial number from odor pred not 108. Execution halted.")
} 

odor_pred_dat = dat %>% 
  mutate(key_resp.keys_raw = gsub("'", "", dat$key_resp.keys_raw)) %>%
  mutate_at(vars(trial, cue_idx, odor_num, correct,
                 ITIDur,odor_name_left,odor_name_center,odor_name_right,
                 key_resp.keys_raw,key_resp.rt_raw), as.numeric) %>%
  mutate(resp = key_resp.keys_raw, 
         rt = key_resp.rt_raw,
         acc = as.numeric(correct==key_resp.keys_raw)) %>%
  select(Run, trial, cue_idx, odor_num, correct,
         ITIDur,odor_name_left,odor_name_center,odor_name_right,
         resp, acc, rt)

### pre-conditioning task ###
# Day 2, 1-3 runs, 24 trials per run
pc_file_name_pattern = 'Sub.*_Day2_Preconditioning_.*Trials\\.csv'
pc_file = list.files(path = path,pattern = pc_file_name_pattern)
if (length(pc_file) != 1) {
  stop("pc_file Day2 not unique. Execution halted.")
} 

dat = read.csv(file.path(path,pc_file))
dat$run = as.numeric(dat$run) # coerce Run col from char to num
dat = subset(dat,run > 0)       # select rows
print(paste('Trial numbers from pc task on Day 2:',nrow(dat)))

pc_dat_Day2 = dat %>% 
  mutate(key_resp.keys_raw = gsub("'", "", dat$key_resp.keys_raw)) %>%
  mutate_at(vars(trial), as.numeric) %>%
  mutate(resp = key_resp.keys_raw, 
         rt = key_resp.rt_raw,
         acc = as.numeric(correct==key_resp.keys_raw)) %>%
  select(run,trial,pair,cue1_idx,cue2_idx,left_idx,right_idx,correct,ITIDur,
         resp,acc,rt) 

# Day 1, 3-5 runs, 24 trials per run
pc_file_name_pattern = 'Sub.*_Day1_Preconditioning_.*Trials\\.csv'
pc_file = list.files(path = path,pattern = pc_file_name_pattern)
if (length(pc_file) != 1) {
  stop("pc_file Day1 not unique. Execution halted.")
} 

dat = read.csv(file.path(path,pc_file))
dat$run = as.numeric(dat$run) # coerce Run col from char to num
dat = subset(dat,run > 0)       # select rows
print(paste('Trial numbers from pc task on Day 1:',nrow(dat)))
pc_dat_Day1 = dat %>% 
  mutate(key_resp.keys_raw = gsub("'", "", dat$key_resp.keys_raw)) %>%
  mutate_at(vars(trial), as.numeric) %>%
  mutate(resp = key_resp.keys_raw, 
         rt = key_resp.rt_raw,
         acc = as.numeric(correct==key_resp.keys_raw)) %>%
  select(run,trial,pair,cue1_idx,cue2_idx,left_idx,right_idx,correct,ITIDur,
         resp,acc,rt) 

### recognition task ###
# 1 run, 36 trials per run
recog_file_name_pattern = 'Sub.*_Day2_Recognition_.*Trials\\.csv'
recog_file = list.files(path = path,pattern = recog_file_name_pattern)
if (length(recog_file) != 1) {
  stop("recog_file not unique. Execution halted.")
} 
dat = read.csv(file.path(path,recog_file))
dat$trial = as.numeric(dat$trial) 
dat = subset(dat,trial > 0)      
if (nrow(dat) != 36) {
  stop("Trial number from recog task not 36. Execution halted.")
} 

recog_dat = dat %>% 
  mutate(key_resp.keys_raw = gsub("'", "", dat$key_resp.keys_raw)) %>%
  mutate_at(vars(pair), as.numeric) %>%
  mutate(resp = key_resp.keys_raw, 
         rt = key_resp.rt_raw,
         acc = as.numeric(correct==key_resp.keys_raw)) %>%
  select(trial,pair,rep,cue1_idx,cue2_idx,correct,ITIDur,
         resp,acc,rt) 

recog_dat %>%
  reframe(mean=mean(acc,na.rm=T))

### independent scan task - character ###
# 4 runs, 32 trials per run
char_file_name_pattern = 'Sub.*_Day1_IndepScan_.*CharTrials\\.csv'
char_file = list.files(path = path,pattern = char_file_name_pattern)
if (length(char_file) != 1) {
  stop("char_file not unique. Execution halted.")
} 
dat = read.csv(file.path(path,char_file))
dat$trial = as.numeric(dat$trial) 
dat = subset(dat,trial > 0)    
if (nrow(dat) != 128) {
  stop("Trial number from char scan not 128. Execution halted.")
} 

char_dat = dat %>% 
  mutate(run = rep(1:4,each=32)) %>%
  mutate(correct = ifelse(correct == "null", "", correct)) %>%
  mutate(key_resp.keys_raw = gsub("'", "", dat$key_resp.keys_raw)) %>%
  mutate_at(vars(rep,cue_idx,ITIDur,key_resp.rt_raw), as.numeric) %>%
  mutate(resp = key_resp.keys_raw, 
         rt = key_resp.rt_raw,
         acc = as.numeric(correct==key_resp.keys_raw)) %>%
  select(run,trial,rep,cue_idx,correct,ITIDur,resp,acc,rt) 

### independent scan task - Place ###
# 4 runs, 32 trials per run
place_file_name_pattern = 'Sub.*_Day1_IndepScan_.*PlaceTrials\\.csv'
place_file = list.files(path = path,pattern = place_file_name_pattern)
if (length(place_file) != 1) {
  stop("place_file not unique. Execution halted.")
} 
dat = read.csv(file.path(path,place_file))
dat$trial = as.numeric(dat$trial) 
dat = subset(dat,trial > 0)      
if (nrow(dat) != 128) {
  stop("Trial number from place scan not 128. Execution halted.")
} 

place_dat = dat %>% 
  mutate(run = rep(1:4,each=32)) %>%
  mutate(correct = ifelse(correct == "null", "", correct)) %>%
  mutate(key_resp_2.keys_raw = gsub("'", "", dat$key_resp_2.keys_raw)) %>%
  mutate_at(vars(rep,cue_idx,ITIDur,key_resp_2.rt_raw), as.numeric) %>%
  mutate(resp = key_resp_2.keys_raw, 
         rt = key_resp_2.rt_raw,
         acc = as.numeric(correct==key_resp_2.keys_raw)) %>%
  select(run,trial,rep,cue_idx,correct,ITIDur,resp,acc,rt) 

# save them to excel files and RData
dat_list = c('char_dat','place_dat',
             'pc_dat_Day1','pc_dat_Day2',
             'pre_meal_dat','post_meal_dat','odor_pred_dat','recog_dat')
for(i in 1:length(dat_list)){
  if (exists(dat_list[i])) {
    tmp_df = get(dat_list[i])
    write.xlsx(tmp_df,file = file.path(outdir,paste0(dat_list[i],'.xlsx')))
    save(tmp_df,file = file.path(outdir,paste0(dat_list[i],'.RData')))
  }
}

} # loop of subjects



