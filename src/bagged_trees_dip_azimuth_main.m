%% Bagged Trees Classifier for Dip Azimuth Prediction (7 features → 12 classes)
% Paper: "Deep learning-based prediction and modeling of natural fractures
%        within ultra-deep reservoirs of Kuqa Depression, Tarim Basin of China"
% Purpose: Train an ensemble of 200 decision trees (Bagged Trees) with
%          cost-sensitive learning to classify fracture dip azimuth into
%          12 categories (30° intervals). The model uses 7 conventional
%          well logs as input features.
%
% NOTE: This script requires MATLAB R2020a or later with Statistics and
%       Machine Learning Toolbox. No interactive input is needed.

clear; clc; close all;

% ==================== User-configurable parameters ====================
data_file       = 'Date_Fracture_orientation_Dip_azimuth.xlsx';   % Training data (7 features + 1 label)
output_dir      = './results_azimuth';           % Output directory
rng_seed        = 42;                            % Random seed for reproducibility
num_trees       = 200;                           % Number of bagged trees (as in paper)
max_splits      = 100;                           % Max splits per tree
cost_scale      = 6;                             % Extra penalty for extreme misclass (Class 1 vs 7)
num_classes     = 12;                            % Number of dip azimuth classes
feature_names   = {'Log_1','Log_2','Log_3','Log_4','Log_5','Log_6','Log_7'}; % Must match column order in file
label_name      = 'Dip_azimuth_class';           % Name of label column (informative only)

% ==================== Initialization ====================
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
rng(rng_seed);

% ==================== 1. Load and prepare data ====================
fprintf('Loading data from: %s\n', data_file);
data = readtable(data_file, 'VariableNamingRule', 'preserve');

% Verify required columns exist
if ~all(ismember(feature_names, data.Properties.VariableNames))
    error('Input file must contain columns: %s', strjoin(feature_names, ', '));
end
if size(data,2) < length(feature_names)+1
    error('Data must have at least %d columns (%d features + 1 label).', ...
          length(feature_names)+1, length(feature_names));
end

% Extract features and labels
X = table2array(data(:, feature_names));
Y = table2array(data(:, end));   % Assume last column is label (1-12)

% Validate labels
if any(Y < 1 | Y > num_classes | mod(Y,1) ~= 0)
    error('Labels must be integers between 1 and %d.', num_classes);
end
Y = categorical(Y, 1:num_classes);
fprintf('Loaded %d samples with %d features.\n', size(X,1), size(X,2));

% ==================== 2. Data cleaning ====================
% Remove rows with NaN in features or undefined categorical labels
valid_idx = all(~isnan(X), 2) & ~isundefined(Y);
X = X(valid_idx, :);
Y = Y(valid_idx);
fprintf('Valid samples after cleaning: %d\n', size(X,1));

% ==================== 3. Z-score normalization ====================
mu = mean(X, 1, 'omitnan');
sigma = std(X, 0, 1, 'omitnan');
sigma(sigma == 0) = 1;          % Avoid division by zero
X_norm = (X - mu) ./ sigma;

% ==================== 4. Build cost matrix (ordinal penalty) ====================
% Diagonal: 0 cost. Off-diagonal: |i-j|, except Class 1 vs 7 cost = 6.
cost_matrix = zeros(num_classes);
for i = 1:num_classes
    for j = 1:num_classes
        if i ~= j
            d = abs(i - j);
            if (i==1 && j==7) || (i==7 && j==1)
                cost_matrix(i,j) = cost_scale;   % 6, as per paper
            else
                cost_matrix(i,j) = d;
            end
        end
    end
end

% ==================== 5. Configure Bagged Tree ensemble ====================
% Use 200 trees, max splits 100, all features considered, surrogate splits on.
tree_template = templateTree(...
    'MaxNumSplits', max_splits, ...
    'NumVariablesToSample', 'all', ...
    'Surrogate', 'on');

% Optional: Enable parallel computing if Parallel Computing Toolbox available
try
    parallel_opt = statset('UseParallel', true);
catch
    parallel_opt = statset('UseParallel', false);
end

% ==================== 6. Train the model ====================
fprintf('Training Bagged Trees ensemble (%d trees)...\n', num_trees);
tic;
model = fitcensemble(X_norm, Y, ...
    'Method', 'Bag', ...
    'Learners', tree_template, ...
    'NumLearningCycles', num_trees, ...
    'Cost', cost_matrix, ...
    'Options', parallel_opt);
train_time = toc;
fprintf('Training completed in %.2f seconds.\n', train_time);

% ==================== 7. Cross-validation evaluation ====================
fprintf('Performing 5-fold stratified cross-validation...\n');
cv = cvpartition(Y, 'KFold', 5, 'Stratify', true);
cv_model = crossval(model, 'CVPartition', cv, 'Options', parallel_opt);
cv_accuracy = 1 - kfoldLoss(cv_model, 'LossFun', 'ClassifError');
fprintf('5-fold CV accuracy: %.2f%%\n', cv_accuracy*100);

% Detailed per-class metrics
[pred_labels, pred_scores] = kfoldPredict(cv_model);
% pred_labels is already categorical if training labels are categorical – no conversion needed
[cm, order] = confusionmat(Y, pred_labels);

% Ensure all metrics are row vectors (1 x num_classes)
precision = diag(cm)' ./ sum(cm, 1);
recall    = diag(cm)' ./ sum(cm, 2)';   % sum(cm,2)' gives row vector
f1        = 2 * (precision .* recall) ./ (precision + recall);
precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1(isnan(f1)) = 0;

fprintf('\nPer-class performance:\n');
disp(table((1:num_classes)', precision(:), recall(:), f1(:), ...
    'VariableNames', {'Class','Precision','Recall','F1'}));

% ==================== 8. Feature importance ====================
imp = predictorImportance(model);
figure('Position', [100 100 600 400]);
barh(imp, 'FaceColor', [0.3 0.7 0.5]);
set(gca, 'YTickLabel', feature_names, 'FontSize', 10);
xlabel('Importance'); title('Feature Importance (Bagged Trees)');
grid on;
saveas(gcf, fullfile(output_dir, 'feature_importance.png'));
close(gcf);

% ==================== 9. Confusion matrix visualizations ====================
% Re-predict on training set for full confusion matrix (optional, but informative)
all_pred = predict(model, X_norm);        % returns categorical – DO NOT re-convert
[cm_full, ~] = confusionmat(Y, all_pred);

% Figure 1: Observation counts
figure('Position', [100 100 800 700]);
imagesc(cm_full); colormap(flipud(gray)); colorbar;
title('Confusion Matrix (Counts)'); xlabel('Predicted'); ylabel('Actual');
set(gca, 'XTick', 1:num_classes, 'YTick', 1:num_classes);
for i = 1:num_classes
    for j = 1:num_classes
        if cm_full(i,j) > 0
            text(j, i, num2str(cm_full(i,j)), ...
                'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'r');
        end
    end
end
saveas(gcf, fullfile(output_dir, 'confusion_counts.png'));
close(gcf);

% Figure 2: True Positive Rate (TPR)
tpr = cm_full ./ sum(cm_full,2) * 100;
tpr(isnan(tpr)) = 0;
figure('Position', [300 100 800 700]);
imagesc(tpr); colormap(parula); colorbar; clim([0 100]);
title('True Positive Rate (%)'); xlabel('Predicted'); ylabel('Actual');
set(gca, 'XTick', 1:num_classes, 'YTick', 1:num_classes);
for i = 1:num_classes
    for j = 1:num_classes
        if tpr(i,j) > 1
            text(j, i, sprintf('%.1f', tpr(i,j)), ...
                'HorizontalAlignment', 'center', 'FontSize', 7);
        end
    end
end
saveas(gcf, fullfile(output_dir, 'tpr_matrix.png'));
close(gcf);

% ==================== 10. Save model and deployment function ====================
% Save trained model + normalization params
model_path = fullfile(output_dir, 'bagged_trees_azimuth.mat');
save(model_path, 'model', 'mu', 'sigma', 'cost_matrix', 'num_classes', ...
    'feature_names', 'num_trees', 'max_splits');
fprintf('Model saved to %s\n', model_path);

% Generate standalone prediction function
deploy_code = {
    'function [pred_labels, scores] = predict_dip_azimuth(new_data_file, varargin)'
    '% predict_dip_azimuth Predict dip azimuth classes using trained Bagged Trees.'
    '%'
    '%   [pred_labels, scores] = predict_dip_azimuth(new_data_file)'
    '%       new_data_file : path to Excel file with 7 feature columns'
    '%                      (Log_1, Log_2, Log_3, Log_4, Log_5, Log_6, Log_7).'
    '%       pred_labels   : predicted classes (1-12).'
    '%       scores        : classification scores (N x 12).'
    '%'
    '%   Optionally specify model file as second argument:'
    '%   predict_dip_azimuth(new_data_file, model_file)'
    ''
    '    if nargin < 1'
    '        error(''Provide path to new data file.'');'
    '    end'
    '    if nargin < 2 || isempty(varargin{1})'
    '        % Use default model in the same folder'
    '        model_file = fullfile(fileparts(mfilename(''fullpath'')), ''bagged_trees_azimuth.mat'');'
    '    else'
    '        model_file = varargin{1};'
    '    end'
    ''
    '    % Load model'
    '    if ~exist(model_file, ''file'')'
    '        error(''Model file not found: %s'', model_file);'
    '    end'
    '    S = load(model_file, ''model'', ''mu'', ''sigma'', ''feature_names'');'
    ''
    '    % Read new data'
    '    new_data = readtable(new_data_file, ''VariableNamingRule'', ''preserve'');'
    '    if ~all(ismember(S.feature_names, new_data.Properties.VariableNames))'
    '        error(''New data must contain columns: %s'', strjoin(S.feature_names, '', ''));'
    '    end'
    '    X = table2array(new_data(:, S.feature_names));'
    ''
    '    % Normalize'
    '    X_norm = (X - S.mu) ./ S.sigma;'
    ''
    '    % Predict'
    '    [pred_labels, scores] = predict(S.model, X_norm);'
    'end'
};
deploy_path = fullfile(output_dir, 'predict_dip_azimuth.m');
fid = fopen(deploy_path, 'w');
fprintf(fid, '%s\n', deploy_code{:});
fclose(fid);
fprintf('Deployment function saved to %s\n', deploy_path);

% ==================== 11. Summary ====================
fprintf('\n=== Training Complete ===\n');
fprintf('Output folder: %s\n', output_dir);
fprintf('  - bagged_trees_azimuth.mat   : trained model + parameters\n');
fprintf('  - predict_dip_azimuth.m      : standalone prediction function\n');
fprintf('  - feature_importance.png     : predictor ranking\n');
fprintf('  - confusion_counts.png       : confusion matrix (counts)\n');
fprintf('  - tpr_matrix.png             : true positive rates\n');
fprintf('  - evaluation in command window.\n');