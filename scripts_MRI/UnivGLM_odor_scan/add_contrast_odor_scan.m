
%% Add contrasts for odor scan GLM
% define contrast names and weights

clear contr_name contr sessrep

% compare odor vs. clean air
contr_name{1} = 'odor>clean-air'; 
contr{1} = [-1,1/2,1/2]; % weights replicated across runs
sessrep{1} = 'repl';

%% SPM contrast
matlabbatch = cell(1,length(contr_name));
for c = 1:length(contr_name)
    matlabbatch{c}.spm.stats.con.spmmat = {fullfile(modeldir,'SPM.mat')};
    matlabbatch{c}.spm.stats.con.consess{1}.tcon.name = contr_name{c};
    matlabbatch{c}.spm.stats.con.consess{1}.tcon.weights = contr{c};
    matlabbatch{c}.spm.stats.con.consess{1}.tcon.sessrep = sessrep{c};
    matlabbatch{c}.spm.stats.con.delete = 0; 
    if c == 1
        matlabbatch{c}.spm.stats.con.delete = 1; % delete existing contrasts if rerun this script
    end
end

%% run the jobs
fname = fullfile(jobdir, sprintf('AddContrasts_%s_%s.mat',modelname,SubID));
save(fname, 'matlabbatch');
spm_jobman('run', matlabbatch);  
