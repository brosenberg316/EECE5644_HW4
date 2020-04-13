function [num_perceptrons, init_weights] =  mlp_model_order_selection(data)
% Returns the optimal number of perceptrons in the first layer 
% for trained MLP of a given dataset

% Range of perceptrons to test
perceptron_range = 1:12;
num_epochs = 6;
% nX = size(data,1);
N = size(data,2);
% 10-fold cross-validation
K = 10;

% Divide dataset into 10 folds
partition_indices = zeros(K,2);
partitions = ceil(linspace(0,N,K+1));
for k = 1:K
    partition_indices(k,:) = [partitions(k)+1, partitions(k+1)];
end
wins = zeros(1,perceptron_range(end));
best_weights = {};
for epoch = 1:num_epochs
    mean_mse = zeros(1,perceptron_range(end));
    % Try all perceptron counts
    for num_perceptrons = 1:perceptron_range(end)
        mse = inf(1,K);
        % Using cross-validation on all folds of dataset
        for k = 1:K
            % Get validation dataset fold
            idx_validate = (partition_indices(k,1):partition_indices(k,2));
            % Using fold k as validation set
            d_validate = data(:,idx_validate);
            
            % Get training dataset, which is rest of data
            idx_train = ~ismember(1:N, idx_validate);
            d_train = data(:,idx_train);
            x1_train = d_train(1,:);
            x2_train = d_train(2,:);
            
            % Initialize the MLP
            net = init_mlp(num_perceptrons,d_train);
            % Train the MLP
            net = train(net,x1_train,x2_train);
            
            % Validate MLP performance against the validation fold
            x1_validate = d_validate(1,:);
            x2_validate = d_validate(2,:);
            outputs = net(x1_validate);
            
            % Keep track of MSE for each fold
            mse(k) = mean((x2_validate-outputs).^2);
           
            % Save the best final weights for each perceptron count
            [~,best_idx] = min(mse);
            if best_idx == k
                best_weights{epoch,num_perceptrons} = {mse(k) net.b{1} net.b{2} net.IW{1} net.LW{2,1}};
            end
        end
        mean_mse(num_perceptrons) = mean(mse,2);
    end
    plot_mse(mean_mse,N,epoch);
    % Find the best num perceptrons
    [~,idx] = min(mean_mse);
    wins(idx) = wins(idx) + 1;
end
% Find the number of perceptrons with the most wins
[~,num_perceptrons] = max(wins);
% Find the trained weights associated with the lowest mse
weights = best_weights(:,num_perceptrons);
[~,idx] = min(cellfun(@(x) x{1},weights));
init_weights = weights{idx}(2:end);
% Print out model selection parameters for reporting
fprintf('Model Order Selection for %d Samples\nNumber of Perceptrons: %d',...
    N,num_perceptrons);