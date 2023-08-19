function[YPred] = Model_LSTM(Train_Data, Train_Target, Test_Data, sol)



numFeatures  = size(Train_Data, 2);
numHiddenUnits = sol;
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
YPred = predict(net,Test_Data,'MiniBatchSize',1);

% actual = Test_Target;
% predicted = YPred;
% actual = Test_Target(3:end);
% Eval = evaluate_error(predicted, actual);
end

