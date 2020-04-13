function [model_order,init_params] = gmm_model_order(data)
% Determines GMM model order (# of clusters) using 10-fold cross-validation
N = length(data); K = 10;
% Divide dataset into 10 folds
partition_indices = zeros(K,2);
partitions = ceil(linspace(0,N,K+1));
for k = 1:K
    partition_indices(k,:) = [partitions(k)+1, partitions(k+1)];
end
perf_vals = zeros(K,6); best_params = {};
% Tweak max number of iterations and convergence tolerance
options = statset('MaxIter',100000,'TolFun',1e-5);
% Using cross-validation on all folds of dataset
for k = 1:K
    % Get validation dataset
    idx_validate = [partition_indices(k,1):partition_indices(k,2)];
    % Using fold k as validation set
    d_validate = data(:,idx_validate);
    % Get training dataset
    idx_train = ~ismember(1:N, idx_validate);
    d_train = data(:,idx_train);
    % Iterate through model orders
    for M = 1:8
        % Perform expectation maximization
        gmm = fitgmdist(d_train',M,'Options',options);
        % Measure performance against validation fold
        alpha = gmm.ComponentProportion;
        mu = gmm.mu';
        Sigma = gmm.Sigma;
        perf_vals(k,M) = measure_performance(alpha,mu,Sigma,d_validate);
        [~,max_perf_idx]  = max(perf_vals(:,M));
        if max_perf_idx == k
            best_params{M} = {alpha,mu,Sigma};
        end
    end
end
% Determine model order with maximum performance value
[~,model_order] = max(mean(perf_vals,1));
init_params = best_params{model_order};
bar(1:M,mean(perf_vals)); xlabel('Number of Clusters'); ylabel('Log-Likelihood');
title('Mean Validation Log-Likelihood for 10-Fold Cross-Validation');