%% run_quick_test.m – Robust quick verification (fixed path replacement)
%  All local functions are placed at the end of the file.
%  Run from the repository root:
%      run('tests/run_quick_test.m')

clear; clc; close all;
disp('========== Quick Test Started ==========');

% Move to repository root
cd(fileparts(mfilename('fullpath')));
cd ..;

% Add src/ to path if it exists
if exist('src', 'dir')
    addpath('src');
end

% Make sure output root exists
if ~exist('outputs', 'dir'), mkdir('outputs'); end

% -------------------- Test 1: CNN Fracture Intensity --------------------
fprintf('\n=== Test 1: CNN Fracture Intensity ===\n');
temp_file = 'cnn_quick.m';
try
    create_test_script('cnn_fdi_main.m', temp_file, ...
        'Date_Fracture_intensity.xlsx', 'intensity');
    run(temp_file);
    fprintf('Test 1 (CNN) completed successfully.\n');
catch ME
    warning('Test 1 (CNN) failed: %s', ME.message);
end
if exist('temp_file','var') && exist(temp_file,'file')
    delete(temp_file);
end

% -------------------- Test 2: Dip Azimuth (Bagged Trees) --------------------
fprintf('\n=== Test 2: Dip Azimuth Classification ===\n');
temp_file = 'az_quick.m';
try
    create_test_script('bagged_trees_dip_azimuth_main.m', temp_file, ...
        'Date_Fracture_orientation_Dip_azimuth.xlsx', 'azimuth');
    run(temp_file);
    fprintf('Test 2 (Dip Azimuth) completed successfully.\n');
catch ME
    warning('Test 2 (Dip Azimuth) failed: %s', ME.message);
end
if exist('temp_file','var') && exist(temp_file,'file')
    delete(temp_file);
end

% -------------------- Test 3: Dip Angle (Bagged Trees) --------------------
fprintf('\n=== Test 3: Dip Angle Classification ===\n');
temp_file = 'ang_quick.m';
try
    create_test_script('bagged_trees_dip_angle_main.m', temp_file, ...
        'Date_Fracture_orientation_Dip_angle.xlsx', 'dip_angle');
    run(temp_file);
    fprintf('Test 3 (Dip Angle) completed successfully.\n');
catch ME
    warning('Test 3 (Dip Angle) failed: %s', ME.message);
end
if exist('temp_file','var') && exist(temp_file,'file')
    delete(temp_file);
end

% -------------------- Done --------------------
fprintf('\n========== Quick Test Finished ==========\n');
fprintf('Results saved in the "outputs/" folder:\n');
fprintf('  outputs/intensity/   (if CNN succeeded)\n');
fprintf('  outputs/azimuth/     (if azimuth classifier succeeded)\n');
fprintf('  outputs/dip_angle/   (if dip angle classifier succeeded)\n');

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function create_test_script(original_script, temp_script, data_file_name, output_subdir)
    % Read the original script
    code = fileread(original_script);
    
    % --------------------------------------------------
    % Fix data_file: replace any path with 'data/filename'
    % --------------------------------------------------
    % 1) If there's a forward or backward slash, capture the filename after it
    code = regexprep(code, ...
        "(data_file\s*=\s*)'[^']*[/\\]([^/\\]+\.xlsx)'", ...
        ['$1''data/$2''']);
    % 2) If there's no slash (pure filename), replace entirely
    code = regexprep(code, ...
        "(data_file\s*=\s*)'([^/\\]+\.xlsx)'", ...
        ['$1''data/$2''']);
    
    % --------------------------------------------------
    % Fix output_dir: force to ./outputs/<subdir>
    % --------------------------------------------------
    code = regexprep(code, ...
        "(output_dir\s*=\s*)'[^']*'", ...
        ['$1''./outputs/' output_subdir '''']);
    
    % --------------------------------------------------
    % Disable well-out validation for quick test
    % --------------------------------------------------
    code = regexprep(code, ...
        "(well_B1_file\s*=\s*)'[^']*'", ...
        '$1''''');
    
    % --------------------------------------------------
    % Fix predict_file (if present): force to data/DYC.xlsx
    % --------------------------------------------------
    if contains(code, 'predict_file')
        code = regexprep(code, ...
            "(predict_file\s*=\s*)'[^']*[/\\]?([^/\\]+\.xlsx)'", ...
            '$1''data/DYC.xlsx''');
        % Also handle the case where it was a pure filename
        code = regexprep(code, ...
            "(predict_file\s*=\s*)'[^']+\.xlsx'", ...
            '$1''data/DYC.xlsx''');
    end
    
    % --------------------------------------------------
    % Reduce epochs to 50 for CNN quick test
    % --------------------------------------------------
    if contains(code, 'epochs')
        code = regexprep(code, ...
            "(epochs\s*=\s*)\d+", ...
            '$150');
    end
    
    % Write temporary script
    fid = fopen(temp_script, 'w');
    fprintf(fid, '%s', code);
    fclose(fid);
end