function features = image_preprocess(image_data)
% Preprocesses image into feature vectors of [row,col,R,G,B]
% Normalizes all vector values into range [0,1]
[R,C,D] = size(image_data);
image_data = double(image_data);
rowIndices = (1:R)'*ones(1,C); colIndices = ones(R,1)*(1:C);
% Initialize feature vectors with [row,col,0,0,0]
features = [rowIndices(:)';colIndices(:)';zeros(3,R*C)];
% Iterate through all colors, add RGB values to each feature vector
for d = 1:D
    color = image_data(:,:,d);
    features(2+d,:) = color(:)';
end
% Normalize the feature vectors
features = normalize(features,2,'range');