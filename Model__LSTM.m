function[Eval,net] = Model__LSTM(Train_Data,  Train_Target,  n, pos)


numFeatures  = size(Train_Data, 2);
numHiddenUnits = 20;
numResponses = 1;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(round(numHiddenUnits),'OutputMode','sequence')
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = 6;
miniBatchSize = 20;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);
net = trainNetwork(Train_Data, Train_Target, layers, options);
net_out = net_1(Train_Data);
rank = sort(net_out, 'descend');
first_N_ranks = rank(1:n);

present1 = find(first_N_ranks == net_out(pos)');

if isempty(present1)
    Accuracy = mean(first_N_ranks)/numel(pos);
    Precision = mean(first_N_ranks)/n;
    MRR =  1/5 * mean(1/rank(1));  % Mean Reciprocal Rank (MRR)
else
    Accuracy = numel(present1)/numel(pos);
    Precision = numel(present1)/n;
    MRR =  1/5 * mean(1/rank(present1(1)));
end

Eval = [Accuracy Precision MRR];

end
