
# code to copy all processed behavioral data 

datDir = '/Users/liuq13/SPCD_data'
outDir = '/Users/liuq13/Library/CloudStorage/Box-Box/SPCD_beh_data'

for(j in 1:42){
  
  SubfolderID = paste0('Sub',j,'_done')
  SubID = paste0('Sub',j)
  path1 = file.path(datDir,SubfolderID,SubID)
  path2 = file.path(datDir,SubfolderID,'beh')
  
  if (dir.exists(path1) & dir.exists(path2)){
    
    print(SubID)
    to_path = file.path(outDir,SubID)
    if (!dir.exists(to_path)) {
      dir.create(to_path)
    }
    file.copy(path1, to_path, recursive = TRUE)
    file.copy(path2, to_path, recursive = TRUE)
    
  }
}

