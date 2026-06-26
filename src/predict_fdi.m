function Y_pred = predict_fdi(new_data_file, model_file, output_file)
% predict_fdi  Predict fracture intensity using a trained CNN model.
%
%   Y_pred = predict_fdi(new_data_file, model_file)
%       new_data_file : string, path to the Excel file containing new well logs.
%                       Columns must match the training data order:
%                       DT, CAL, CNL, GR, DEN, RT, RM (no target column needed).
%       model_file    : string, path to the saved .mat file containing 'net',
%                       'PS_X', and 'PS_Y'.
%       Y_pred        : vector of predicted fracture intensity values.
%
%   Y_pred = predict_fdi(new_data_file, model_file, output_file)
%       output_file   : (optional) string, name of CSV file to save predictions.
%                       If provided, results are written to this file.
%
%   Example:
%       pred = predict_fdi('new_well.xlsx', 'results/cnn_model.mat', 'predicted_fdi.csv');

    % --- Check input arguments ---
    if nargin < 2
        error('At least two arguments required: predict_fdi(new_data_file, model_file, [output_file])');
    end
    if nargin < 3
        output_file = '';
    end

    % --- Load model and normalization parameters ---
    if ~exist(model_file, 'file')
        error('Model file not found: %s', model_file);
    end
    loaded = load(model_file, 'net', 'PS_X', 'PS_Y');
    net = loaded.net;
    PS_X = loaded.PS_X;
    PS_Y = loaded.PS_Y;
    fprintf('Model loaded from: %s\n', model_file);

    % --- Load new data ---
    if ~exist(new_data_file, 'file')
        error('New data file not found: %s', new_data_file);
    end
    new_data = readtable(new_data_file, 'VariableNamingRule', 'preserve');
    X_new = table2array(new_data);

    % Determine number of features from normalization parameters
    expected_features = PS_X.xrows;
    if size(X_new, 2) ~= expected_features
        error(['Mismatch in number of features. Expected %d, got %d. ', ...
               'Check input columns.'], expected_features, size(X_new,2));
    end

    % --- Normalize using saved parameters ---
    X_new_norm = mapminmax('apply', X_new', PS_X);

    % --- Reshape for CNN (4D array) ---
    X_new_4d = reshape(X_new_norm, size(X_new_norm,1), 1, 1, []);

    % --- Predict ---
    Y_pred_norm = predict(net, X_new_4d);
    Y_pred = mapminmax('reverse', Y_pred_norm', PS_Y)';

    % --- Optionally save predictions to CSV ---
    if ~isempty(output_file)
        pred_table = table(Y_pred, 'VariableNames', {'Predicted_FDI'});
        writetable(pred_table, output_file);
        fprintf('Predictions saved to: %s\n', output_file);
    end

    fprintf('Prediction complete. %d samples processed.\n', length(Y_pred));
end