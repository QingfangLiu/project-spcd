
% this code runs independently to copy files in compliance with BIDS format


% Set your base directory here
baseDir = '/Users/liuq13/SPCD_data/nii';
BIDSDir = '/Users/liuq13/SPCD_data_BIDS';

subjectDirs = dir(fullfile(baseDir, 'sub-*'));

%%

for i = [5]  
    
    origSub = subjectDirs(i).name;
    subjNum = regexp(origSub, '\d+', 'match');
    bidsSub = sprintf('sub-%02d', str2double(subjNum{1}));
    
    % Create BIDS folder
    bidsFuncDir = fullfile(BIDSDir, bidsSub, 'func');
    if ~exist(bidsFuncDir, 'dir')
        mkdir(bidsFuncDir);
    end
    
    % Path to raw functional data
    funcRawDir = fullfile(baseDir, origSub, 'func_raw');
    taskDirs = dir(funcRawDir);
    
    for j = 1:length(taskDirs)
        taskName = taskDirs(j).name;
        taskPath = fullfile(funcRawDir, taskName);
        
        if taskDirs(j).isdir && ~strcmp(taskName, '.') && ~strcmp(taskName, '..')
            files = dir(fullfile(taskPath, '*e*.nii*'));  % Look for echo files

            for k = 1:length(files)
                fname = files(k).name;
                
                % Modified to also extract series number (after the timestamp)
                % Example: Scan2_premeal_run_3_20250412101822_15_e1.nii.gz
                tokens = regexp(fname, '.*_([\w_]+)_run_(\d+)_\d+_(\d+)_e(\d+)\.nii(\.gz)?', 'tokens');
                
                info = tokens{1};
                rawTask = info{1};
                run = info{2};
                series = info{3};  % NEW: sequence number (like 15 or 17)
                echo = info{4};
                
                % Special handling for subject 36 (interrupted run)
                % === Special case ===
                % For subject 36, task "premeal" run-3 was interrupted and saved as two sequences:
                % - Sequence 15 → treated as run-3
                % - Sequence 17 → treated as run-4
                % This is handled in the parsing logic using the extracted sequence number.

                if str2double(subjNum{1}) == 36 & strcmp(rawTask,'premeal')
                    if strcmp(series, '15')
                        run = '3';
                    elseif strcmp(series, '17')
                        run = '4';
                    end
                end

                % === Task name mapping ===
                taskMap = containers.Map( ...
                    {'scan', 'odor_scan'}, ...   % raw task names in your filenames
                    {'odorscan', 'odorscan'});   % desired BIDS-compliant names
                
                % Use mapped task name if found, otherwise use raw
                if isKey(taskMap, rawTask)
                    task = taskMap(rawTask);
                else
                    task = strrep(rawTask, '_', '');  % fallback: remove underscores
                end

                % New filename
                newName = sprintf('%s_task-%s_run-%s_echo-%s_bold.nii.gz', bidsSub, task, run, echo);
                sourceFile = fullfile(taskPath, fname);
                destFile = fullfile(bidsFuncDir, newName);

                % Copy file
                copyfile(sourceFile, destFile);

                % also copy json files
                sourceFile_json = strrep(sourceFile, "nii.gz", "json");
                destFile_json = strrep(destFile, "nii.gz", "json");
                copyfile(sourceFile_json, destFile_json)

                
                % Append "TaskName" to existing JSON
                % Load, append, and rewrite
                txt = fileread(destFile_json);
                txt = strtrim(txt);
    
                if endsWith(txt, '}')
                    txt = txt(1:end-1); % Remove closing brace
                    if ~endsWith(txt, ',')
                        txt = [txt, ','];
                    end
                    txt = sprintf('%s\n  "TaskName": "%s"\n}', txt, task);
            
                    fid = fopen(destFile_json, 'w');
                    fwrite(fid, txt);
                    fclose(fid);
                end

            end
        end
    end % end of task dirs


    % === Anatomical directory ===
    anatRawDir = fullfile(baseDir, origSub, 'anat');
    bidsAnatDir = fullfile(BIDSDir, bidsSub, 'anat');
    if ~exist(bidsAnatDir, 'dir')
        mkdir(bidsAnatDir);
    end

    anatFiles = dir(fullfile(anatRawDir, '*_defaced.nii'));
    for a = 1:length(anatFiles)
        anatFname = anatFiles(a).name;
        anatDestName = sprintf('%s_T1w.nii', bidsSub);
        anatSourceFile = fullfile(anatRawDir, anatFname);
        anatDestFile = fullfile(bidsAnatDir, anatDestName);
        copyfile(anatSourceFile, anatDestFile);
        system(['gzip "' anatDestFile '"']);
    end



    %% labchart data
% srcDir = fullfile(baseDir, origSub, 'labchart'); 
% destBase = fullfile(BIDSDir, 'derivatives', 'labchart_processed');
% 
% % Files to copy (only Day1 and Day2, .mat and .bmp)
% fileList = dir(fullfile(srcDir, '*.*'));
% validExts = {'.mat', '.bmp'};
% 
% % Loop through files
% for k = 1:length(fileList)
%     fileName = fileList(k).name;
% 
%     [~, ~, ext] = fileparts(fileName);
% 
%     % Check file extension
%     if ~ismember(lower(ext), validExts)
%         continue;
%     end
% 
%     % Determine day
%     if contains(fileName, 'Day1')
%         day = 'day1';
%     elseif contains(fileName, 'Day2')
%         day = 'day2';
%     else
%         continue; % Skip anything else
%     end
% 
%     % Create destination folder if it doesn't exist
%     destDir = fullfile(destBase, bidsSub, day);
%     if ~exist(destDir, 'dir')
%         mkdir(destDir);
%     end
% 
%     % Copy file
%     srcPath = fullfile(srcDir, fileName);
%     destPath = fullfile(destDir, fileName);
%     copyfile(srcPath, destPath);
%     fprintf('Copied %s to %s\n', fileName, destPath);
% end




    

end % end of subject folders
