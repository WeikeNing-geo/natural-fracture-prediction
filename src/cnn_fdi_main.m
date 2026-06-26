%% CNN Fracture Intensity Prediction – Main Script (Proportional Split + Visualizations)
% Paper: "Deep learning-based prediction and modeling of natural fractures
%        within ultra-deep reservoirs of Kuqa Depression, Tarim Basin of China"
% Purpose: Train a 1D CNN to predict fracture density index (FDI) from
%          7 conventional well logs, with independent validation & test sets,
%          well-out validation on Well B1, and automatic figure generation.
%
% NOTE: This script contains NO interactive inputs and uses a fixed random
%       seed to ensure reproducibility.

clear; clc; close all;

% ==================== User-configurable parameters ====================
data_file       = 'Date_Fracture_intensity.xlsx';   % Training data (exclude B1)
predict_file    = 'DYC.xlsx';                        % Optional: new well data for prediction
well_B1_file    = 'B1_well_data.xlsx';               % Independent B1 well for well-out validation
output_dir      = './results';                       % Output directory for results
train_ratio     = 0.8;                               % Proportion of training set
val_ratio       = 0.1;                               % Proportion of validation set (rest for test)
epochs          = 1500;                              % Maximum epochs
mini_batch      = 128;                               % Mini-batch size
initial_lr      = 1e-4;                              % Initial learning rate
lr_drop_factor  = 0.7;                               % Learning rate drop factor
lr_drop_period  = 300;                               % Drop period (in epochs)
l2_reg          = 0.0005;                            % L2 regularization coefficient
gradient_clip   = 1;                                 % Gradient clipping threshold
val_freq        = 100;                               % Validation frequency (iterations)
rng_seed        = 42;                                % Random seed for reproducibility

% Create output directory if it doesn't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
rng(rng_seed);   % Fix random seed

% ==================== 1. Load and clean data ====================
fprintf('Loading training data: %s\n', data_file);
data = readtable(data_file, 'VariableNamingRule', 'preserve');
input_names = data.Properties.VariableNames(1:end-1);
target_name = data.Properties.VariableNames{end};

% Convert to matrices
X = table2array(data(:, 1:end-1));
Y = table2array(data(:, end));

% Basic cleaning: remove NaN and outliers (adjust range as needed)
valid_idx = all(~isnan(X), 2) & ~isnan(Y) & ...
            all(X > -100 & X < 1000, 2) & (Y > -100 & Y < 1000);
X = X(valid_idx, :);
Y = Y(valid_idx);
fprintf('Valid samples: %d\n', size(X, 1));

% Ensure the number of features is 7 (DT, CAL, CNL, GR, DEN, RT, RM)
if size(X, 2) ~= 7
    warning(['Number of input features is %d, not 7. ', ...
             'Check if columns match the selected logs in the paper.'], size(X,2));
end

% ==================== 2. Split data into training, validation, and test sets ====================
n_total = size(X, 1);
n_train = round(n_total * train_ratio);
n_val   = round(n_total * val_ratio);
n_test  = n_total - n_train - n_val;

if n_train < 50 || n_val < 50 || n_test < 50
    error('Each split must contain at least 50 samples. Adjust ratios or check your data.');
end

% Random permutation
idx = randperm(n_total);
X_train = X(idx(1:n_train), :);
Y_train = Y(idx(1:n_train));
X_val   = X(idx(n_train+1 : n_train+n_val), :);
Y_val   = Y(idx(n_train+1 : n_train+n_val));
X_test  = X(idx(n_train+n_val+1 : end), :);
Y_test  = Y(idx(n_train+n_val+1 : end));

fprintf('Training: %d, Validation: %d, Test: %d\n', n_train, n_val, n_test);

% ==================== 3. Normalize data using Min-Max scaling ====================
[X_train_norm, PS_X] = mapminmax(X_train', 0, 1);
[Y_train_norm, PS_Y] = mapminmax(Y_train', 0, 1);

X_val_norm   = mapminmax('apply', X_val', PS_X);
Y_val_norm   = mapminmax('apply', Y_val', PS_Y);
X_test_norm  = mapminmax('apply', X_test', PS_X);
Y_test_norm  = mapminmax('apply', Y_test', PS_Y);

% Reshape to 4-D arrays required by CNN: [features, 1, 1, samples]
X_train_4d = reshape(X_train_norm, size(X_train_norm,1), 1, 1, []);
Y_train_4d = Y_train_norm';
X_val_4d   = reshape(X_val_norm,   size(X_val_norm,1),   1, 1, []);
Y_val_4d   = Y_val_norm';
X_test_4d  = reshape(X_test_norm,  size(X_test_norm,1),  1, 1, []);
Y_test_4d  = Y_test_norm';

% ==================== 4. Build 1D CNN architecture (as in Table 1) ====================
layers = [
    imageInputLayer([size(X_train_norm,1), 1, 1], 'Name', 'input')
    
    % --- First convolutional block ---
    convolution2dLayer([3,1], 64, 'Padding', 'same', 'Name', 'conv1_1')
    batchNormalizationLayer('Name', 'bn1_1')
    leakyReluLayer(0.01, 'Name', 'lrelu1_1')
    convolution2dLayer([3,1], 64, 'Padding', 'same', 'Name', 'conv1_2')
    batchNormalizationLayer('Name', 'bn1_2')
    leakyReluLayer(0.01, 'Name', 'lrelu1_2')
    maxPooling2dLayer([2,1], 'Stride', 2, 'Name', 'pool1')
    
    % --- Second convolutional block ---
    convolution2dLayer([3,1], 128, 'Padding', 'same', 'Name', 'conv2_1')
    batchNormalizationLayer('Name', 'bn2_1')
    leakyReluLayer(0.01, 'Name', 'lrelu2_1')
    convolution2dLayer([3,1], 128, 'Padding', 'same', 'Name', 'conv2_2')
    batchNormalizationLayer('Name', 'bn2_2')
    leakyReluLayer(0.01, 'Name', 'lrelu2_2')
    maxPooling2dLayer([2,1], 'Stride', 2, 'Name', 'pool2')
    
    % --- Third convolutional block (no pooling) ---
    convolution2dLayer([3,1], 256, 'Padding', 'same', 'Name', 'conv3_1')
    batchNormalizationLayer('Name', 'bn3_1')
    leakyReluLayer(0.01, 'Name', 'lrelu3_1')
    convolution2dLayer([3,1], 256, 'Padding', 'same', 'Name', 'conv3_2')
    batchNormalizationLayer('Name', 'bn3_2')
    leakyReluLayer(0.01, 'Name', 'lrelu3_2')
    
    % --- Channel adjustment with 1x1 convolution ---
    convolution2dLayer([1,1], 512, 'Padding', 'same', 'Name', 'conv4_1')
    batchNormalizationLayer('Name', 'bn4_1')
    leakyReluLayer(0.01, 'Name', 'lrelu4_1')
    
    % --- Fourth convolutional block ---
    convolution2dLayer([3,1], 512, 'Padding', 'same', 'Name', 'conv4_2')
    batchNormalizationLayer('Name', 'bn4_2')
    leakyReluLayer(0.01, 'Name', 'lrelu4_2')
    convolution2dLayer([3,1], 512, 'Padding', 'same', 'Name', 'conv4_3')
    batchNormalizationLayer('Name', 'bn4_3')
    leakyReluLayer(0.01, 'Name', 'lrelu4_3')
    
    % --- Global average pooling ---
    globalAveragePooling2dLayer('Name', 'gap')
    
    % --- Fully connected layers with dropout ---
    fullyConnectedLayer(1024, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn_fc1')
    leakyReluLayer(0.01, 'Name', 'lrelu_fc1')
    dropoutLayer(0.6, 'Name', 'drop1')
    
    fullyConnectedLayer(512, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn_fc2')
    leakyReluLayer(0.01, 'Name', 'lrelu_fc2')
    dropoutLayer(0.5, 'Name', 'drop2')
    
    fullyConnectedLayer(256, 'Name', 'fc3')
    batchNormalizationLayer('Name', 'bn_fc3')
    leakyReluLayer(0.01, 'Name', 'lrelu_fc3')
    dropoutLayer(0.4, 'Name', 'drop3')
    
    fullyConnectedLayer(128, 'Name', 'fc4')
    batchNormalizationLayer('Name', 'bn_fc4')
    leakyReluLayer(0.01, 'Name', 'lrelu_fc4')
    dropoutLayer(0.3, 'Name', 'drop4')
    
    fullyConnectedLayer(1, 'Name', 'output')
    regressionLayer('Name', 'regression')
];

% ==================== 5. Specify training options ====================
options = trainingOptions('adam', ...
    'MiniBatchSize', mini_batch, ...
    'MaxEpochs', epochs, ...
    'InitialLearnRate', initial_lr, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', lr_drop_factor, ...
    'LearnRateDropPeriod', lr_drop_period, ...
    'L2Regularization', l2_reg, ...
    'GradientThreshold', gradient_clip, ...
    'ValidationData', {X_val_4d, Y_val_4d}, ...
    'ValidationFrequency', val_freq, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 50);

% ==================== 6. Train the network ====================
fprintf('Training CNN...\n');
[net, info] = trainNetwork(X_train_4d, Y_train_4d, layers, options);

% ==================== 7. Evaluate on training, validation, and test sets ====================
Y_train_pred_norm = predict(net, X_train_4d);
Y_train_pred = mapminmax('reverse', Y_train_pred_norm', PS_Y)';

Y_val_pred_norm = predict(net, X_val_4d);
Y_val_pred = mapminmax('reverse', Y_val_pred_norm', PS_Y)';

Y_test_pred_norm = predict(net, X_test_4d);
Y_test_pred = mapminmax('reverse', Y_test_pred_norm', PS_Y)';

% Compute metrics
train_rmse = sqrt(mean((Y_train - Y_train_pred).^2));
val_rmse   = sqrt(mean((Y_val   - Y_val_pred).^2));
test_rmse  = sqrt(mean((Y_test  - Y_test_pred).^2));

train_r2 = 1 - sum((Y_train - Y_train_pred).^2) / sum((Y_train - mean(Y_train)).^2);
val_r2   = 1 - sum((Y_val   - Y_val_pred).^2) / sum((Y_val   - mean(Y_val)).^2);
test_r2  = 1 - sum((Y_test  - Y_test_pred).^2) / sum((Y_test  - mean(Y_test)).^2);

fprintf('--- Evaluation Results ---\n');
fprintf('Train: RMSE = %.4f, R2 = %.4f\n', train_rmse, train_r2);
fprintf('Val:   RMSE = %.4f, R2 = %.4f\n', val_rmse, val_r2);
fprintf('Test:  RMSE = %.4f, R2 = %.4f\n', test_rmse, test_r2);

% Save evaluation results to CSV
results_table = table({'Train'; 'Val'; 'Test'}, ...
    [train_rmse; val_rmse; test_rmse], ...
    [train_r2; val_r2; test_r2], ...
    'VariableNames', {'Dataset', 'RMSE', 'R2'});
writetable(results_table, fullfile(output_dir, 'evaluation_metrics.csv'));

% ==================== 8. Generate comparison figures ====================
fprintf('Generating comparison figures...\n');

% 8.1 RMSE and R² bar charts
figure('Position', [100, 100, 800, 400]);
subplot(1,2,1);
bar([train_rmse, val_rmse, test_rmse], 'FaceColor', [0.2, 0.6, 1]);
set(gca, 'XTickLabel', {'Train', 'Val', 'Test'});
ylabel('RMSE (fractures/m)'); title('RMSE Comparison'); grid on;

subplot(1,2,2);
bar([train_r2, val_r2, test_r2], 'FaceColor', [0.2, 0.8, 0.4]);
set(gca, 'XTickLabel', {'Train', 'Val', 'Test'});
ylabel('R²'); title('R² Comparison'); grid on;
saveas(gcf, fullfile(output_dir, 'metrics_bar.png'));
fprintf('   -> metrics_bar.png saved\n');

% 8.2 Combined scatter plots for all sets
figure('Position', [100, 100, 900, 350]);
subplot(1,3,1);
scatter(Y_train, Y_train_pred, 15, 'b', 'filled'); hold on;
plot([min(Y_train), max(Y_train)], [min(Y_train), max(Y_train)], 'r--');
xlabel('Actual FDI'); ylabel('Predicted FDI');
title(sprintf('Train (R²=%.3f)', train_r2)); grid on; axis equal;

subplot(1,3,2);
scatter(Y_val, Y_val_pred, 15, 'g', 'filled'); hold on;
plot([min(Y_val), max(Y_val)], [min(Y_val), max(Y_val)], 'r--');
xlabel('Actual FDI'); ylabel('Predicted FDI');
title(sprintf('Validation (R²=%.3f)', val_r2)); grid on; axis equal;

subplot(1,3,3);
scatter(Y_test, Y_test_pred, 15, 'm', 'filled'); hold on;
plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--');
xlabel('Actual FDI'); ylabel('Predicted FDI');
title(sprintf('Test (R²=%.3f)', test_r2)); grid on; axis equal;
sgtitle('Predicted vs Actual Fracture Intensity');
saveas(gcf, fullfile(output_dir, 'scatter_all_sets.png'));
fprintf('   -> scatter_all_sets.png saved\n');

% 8.3 Training loss curves
figure('Position', [100, 100, 600, 400]);
plot(info.TrainingLoss, 'b-', 'LineWidth', 1.2); hold on;
if isfield(info, 'ValidationLoss')
    plot(info.ValidationLoss, 'r-', 'LineWidth', 1.2);
    legend('Training Loss', 'Validation Loss', 'Location', 'northeast');
else
    legend('Training Loss', 'Location', 'northeast');
end
xlabel('Iteration'); ylabel('Loss'); title('Training Progress'); grid on;
saveas(gcf, fullfile(output_dir, 'loss_curve.png'));
fprintf('   -> loss_curve.png saved\n');

% 8.4 Residual histogram (test set)
residuals = Y_test - Y_test_pred;
figure('Position', [100, 100, 500, 400]);
histogram(residuals, 30, 'FaceColor', [0.6, 0.4, 0.8]);
xlabel('Residual (Actual - Predicted)'); ylabel('Frequency');
title(sprintf('Test Set Residuals (RMSE=%.3f)', test_rmse)); grid on;
saveas(gcf, fullfile(output_dir, 'residual_hist.png'));
fprintf('   -> residual_hist.png saved\n');

% ==================== 9. Save trained model and normalization parameters ====================
save(fullfile(output_dir, 'cnn_model.mat'), 'net', 'PS_X', 'PS_Y');
fprintf('Model saved to %s\n', fullfile(output_dir, 'cnn_model.mat'));

% ==================== 10. Well-out validation on B1 well ====================
if exist(well_B1_file, 'file')
    fprintf('Performing well-out validation on %s...\n', well_B1_file);
    b1_data = readtable(well_B1_file, 'VariableNamingRule', 'preserve');
    X_b1 = table2array(b1_data(:, 1:end-1));
    Y_b1_actual = table2array(b1_data(:, end));
    
    X_b1_norm = mapminmax('apply', X_b1', PS_X);
    X_b1_4d = reshape(X_b1_norm, size(X_b1_norm,1), 1, 1, []);
    Y_b1_pred_norm = predict(net, X_b1_4d);
    Y_b1_pred = mapminmax('reverse', Y_b1_pred_norm', PS_Y)';
    
    b1_rmse = sqrt(mean((Y_b1_actual - Y_b1_pred).^2));
    b1_r2   = 1 - sum((Y_b1_actual - Y_b1_pred).^2) / sum((Y_b1_actual - mean(Y_b1_actual)).^2);
    fprintf('Well B1: RMSE = %.4f, R2 = %.4f\n', b1_rmse, b1_r2);
    
    save(fullfile(output_dir, 'well_B1_validation.mat'), ...
        'Y_b1_actual', 'Y_b1_pred', 'b1_rmse', 'b1_r2');
    
    % B1 scatter plot
    figure;
    scatter(Y_b1_actual, Y_b1_pred, 20, 'filled');
    hold on;
    plot([min(Y_b1_actual), max(Y_b1_actual)], ...
         [min(Y_b1_actual), max(Y_b1_actual)], 'r--', 'LineWidth', 1.5);
    xlabel('Actual FDI (B1)'); ylabel('Predicted FDI (B1)');
    title(sprintf('Well-Out Validation (B1): R² = %.3f, RMSE = %.3f', b1_r2, b1_rmse));
    grid on;
    saveas(gcf, fullfile(output_dir, 'B1_validation_scatter.png'));
else
    fprintf('B1 well file not found. Skipping well-out validation.\n');
end

% ==================== 11. Predict on new well data (optional) ====================
if exist(predict_file, 'file')
    fprintf('Predicting on new data: %s\n', predict_file);
    new_data = readtable(predict_file, 'VariableNamingRule', 'preserve');
    X_new = table2array(new_data(:, input_names));
    
    if size(X_new,2) ~= size(X_train,2)
        error('Number of features in prediction file does not match training data.');
    end
    
    X_new_norm = mapminmax('apply', X_new', PS_X);
    X_new_4d = reshape(X_new_norm, size(X_new_norm,1), 1, 1, []);
    Y_new_pred_norm = predict(net, X_new_4d);
    Y_new_pred = mapminmax('reverse', Y_new_pred_norm', PS_Y)';
    
    pred_table = table(Y_new_pred, 'VariableNames', {'Predicted_FDI'});
    writetable(pred_table, fullfile(output_dir, 'predicted_fdi.csv'));
    fprintf('Prediction saved to %s\n', fullfile(output_dir, 'predicted_fdi.csv'));
else
    fprintf('Prediction file not found. Skipping new well prediction.\n');
end

fprintf('All done. Results saved in %s\n', output_dir);