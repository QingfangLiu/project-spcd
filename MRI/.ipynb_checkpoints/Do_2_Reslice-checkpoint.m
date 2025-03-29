
spm('Defaults','fMRI')
global defaults
defaults.stats.maxmem = 2 ^ 34; % based on RAM = 32GB
defaults.stats.resmem = true; % allow temp files to be kept in memory

%%
% organize echo 1 images across runs and sessions
% note different runs have different volume numbers
n_e1 = dir(fullfile(SubDir,'nii', 'func', '*', 'Scan*_e1.nii'));  % raw nii files start from 'Scan'
nruns = length(n_e1);

% count # volumes in each run
nscans = zeros(nruns,1);
for r = 1:nruns
    tmp_file = fullfile(n_e1(r).folder, n_e1(r).name);
    nscans(r) = length(spm_vol(tmp_file));
end

for r = 1:nruns
    
    tmp_folder = n_e1(r).folder;
    tmp_file_name = n_e1(r).name;
    
    
    fprintf('Working on %s\n',tmp_folder)
    
    % reslice volumes in all echoes
    fimages = [];
    rfilename = [];
    func_ctr = 0;
    rfile_by_echo = [];
    
    for iecho = 1:nechos
        curr_echo_file_name = strrep(tmp_file_name, 'e1', ['e', num2str(iecho)]);        
        for i = 1:nscans(r)
            func_ctr = func_ctr + 1;
            fimages{func_ctr,1} = sprintf('%s,%01d', fullfile(tmp_folder, curr_echo_file_name),i);
            rfilename{iecho}{i,1} = sprintf('%s,%01d', fullfile(tmp_folder, sprintf('r%s',curr_echo_file_name)),i);
        end
        
        rfile_by_echo{iecho,1} = fullfile(tmp_folder, sprintf('r%s',curr_echo_file_name));
        
    end

    matlabbatch = [];
    matlabbatch{1}.spm.spatial.realign.write.data = fimages;
    matlabbatch{1}.spm.spatial.realign.write.roptions.which = [2 0]; % reslice all images, but no need to create the mean
    matlabbatch{1}.spm.spatial.realign.write.roptions.interp = 4;
    matlabbatch{1}.spm.spatial.realign.write.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.realign.write.roptions.mask = 1;
    matlabbatch{1}.spm.spatial.realign.write.roptions.prefix = 'r';

    % save and run job
    fname = fullfile(jobdir, sprintf('me_reslice_all_echoes_run_%d.mat',r));
    save(fname, 'matlabbatch');
    spm_jobman('run', matlabbatch);
    
    
    %% COMBINE ECHOS
% compute tSNR maps per echo
tSNPmapsdir = fullfile(SubDir,'nii','tSNPmaps');
if ~exist(tSNPmapsdir,'dir')
    mkdir(tSNPmapsdir)
end

hdr2 = spm_vol(fimages{1,1}); % get one header
hdr2.dt = [16,0];
tSNR = [];
tSNR_TE = [];

for j = 1:nechos
    V = spm_vol(rfile_by_echo{j});  
    data = spm_read_vols(V);  
    mean_signal = mean(data, 4);  % Mean across time
    std_signal = std(data, 0, 4);  % Standard deviation across time
    
    tSNR{j} = mean_signal ./ std_signal;
    
    hdr2.fname = fullfile(tSNPmapsdir, sprintf('tSNR_echo_%d_run%d.nii', j,r));
    spm_write_vol(hdr2, tSNR{j});
    
    % multiply tSNR by TE for each echo
    tSNR_TE(:,:,:,j) = tSNR{j}.*TEs(j);
end


w_tSNRdir = fullfile(SubDir,'nii','w_tSNR');
if ~exist(w_tSNRdir,'dir')
    mkdir(w_tSNRdir)
end
    
% compute weight (w)
w = [];
for j = 1:nechos

    % normalize tSNR*TE maps by sum of tSNR*TE's across echos
    w{j} = tSNR_TE(:,:,:,j)./sum(tSNR_TE,4);

    % save w_tSNR map
    hdr2.fname = fullfile(w_tSNRdir, sprintf('w_tSNR_TE_echo_%d_run%d.nii', j,r));
    spm_write_vol(hdr2, w{j});
end


functdirPAID = strrep(tmp_folder, 'func', 'func_PAID');   
if ~exist(functdirPAID,'dir')
    mkdir(functdirPAID)
end

% combine echos using w
for i = 1:nscans(r)
    y = zeros(size(w{j}));
    for j=1:nechos
        hdr = spm_vol(rfilename{j}{i,1});
        x = spm_read_vols(hdr);
        y = y + (x.*w{j});
    end

    % save combined volume
    hdr.n = [1 1];
    hdr.fname = fullfile(functdirPAID, sprintf('fvol_%03d.nii', i));
    spm_write_vol(hdr, y);
end

% convert 3D images to 4d images and rename
image_4D_name = strrep(tmp_file_name, 'e1', 'PAID');
func_4d = fullfile(functdirPAID, image_4D_name);
    
if ~exist(func_4d,'file')
    disp('Working on 3D to 4D ...')
    func_files_list = dir(fullfile(functdirPAID,'fvol*.nii'));
    func_files = arrayfun(@(x) fullfile(functdirPAID, x.name), func_files_list, 'UniformOutput', false);
    spm_file_merge(func_files, func_4d, 0);  % Merge 3D volumes into a 4D volume
    for i = 1:numel(func_files)
        delete(func_files{i});
    end
end


end