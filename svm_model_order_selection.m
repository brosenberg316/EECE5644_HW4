function [C, Sigma] =  svm_model_order_selection(data,labels)
% Returns the optimal contraint violation hyperparameter C, and 
% kernel width parameter Sigma for a SVM with Gaussian kernel

% Range of constraint parameters to test
C_range = logspace(-2,1,25); C_len = length(C_range);
% Range of kernel width parameters to test
Sigma_range = logspace(-1,2,25);

% 10-fold cross-validation
K = 10; N = size(data,2);
% Divide dataset into 10 folds
partition_indices = zeros(K,2);
partitions = ceil(linspace(0,N,K+1));
for k = 1:K
    partition_indices(k,:) = [partitions(k)+1, partitions(k+1)];
end

p_correct = zeros(length(C_range),length(Sigma_range));
% Try all width parameters in range
for s = 1:length(Sigma_range)
    sigma = Sigma_range(s);
    % Try all constraint parameters in range
    for c = 1:C_len
        num_correct = zeros(1,K);
        C = C_range(c);
        % Using cross-validation on all folds of dataset
        for k = 1:K
            % Get validation dataset fold
            idx_validate = (partition_indices(k,1):partition_indices(k,2));
            % Using fold k as validation set
            d_validate = data(:,idx_validate);
            l_validate = labels(idx_validate);
            
            % Get training dataset, which is rest of data
            idx_train = ~ismember(1:N, idx_validate);
            d_train = data(:,idx_train);
            l_train = labels(idx_train);
            
            % Train the SVM
            trainedSVM = fitcsvm(d_train',l_train,...
                'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);
            
            % Validate SVM performance against the validation fold
            y = trainedSVM.predict(d_validate');
            
            % Keep track of number correct for each fold
            num_correct(k) = sum(l_validate == y');          
        end
        % Keep track of p(correct) for each C,Sigma combination
        p_correct(c,s) = sum(num_correct)/N;
    end
    % plot_p_errors(mean_p_error,N);
end
% Find the C,Sigma combo with the highest P(correct)
[~,idx] = max(p_correct,[],'all','linear');
[C_idx,Sigma_idx] = ind2sub([length(C_range) length(Sigma_range)],idx);
C = C_range(C_idx); Sigma = Sigma_range(Sigma_idx);
% Print out parameters for reporting
fprintf('Best SVM Parameters for %d Samples\nC: %f\nSigma: %f',N,C,Sigma);

figure()
contour(log10(C_range),log10(Sigma_range),p_correct'); 
xlabel('log_{10} C'); ylabel('log_{10} \sigma');
title({'Gaussian Kernel SVM', '10-fold Cross-Validation Accuracy Estimate'}), axis equal,