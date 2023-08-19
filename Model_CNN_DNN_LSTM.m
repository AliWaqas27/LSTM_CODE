function[Eval] = Model_CNN_DNN_LSTM(Feat,All_Scores, n, pos)


fetaureWeight = Feat;


% Deep Neural Network (DNN)
net = feedforwardnet([5 3]);
Feat = double(fetaureWeight);
Tar = All_Scores;
net_1 = train(net,Feat',Tar);               % train data
net_out1 = net_1(Feat');
rank1 = sort(net_out1, 'descend');
first_N_ranks1 = rank1(1:n);
% LSTM
net_out2 = Model_LSTM(Feat',Tar, Feat', 15);
rank2 = sort(net_out2, 'descend');
rank = (rank1+rank2)/2;
first_N_ranks2 = rank2(1:n);
present1 = find(first_N_ranks1 == net_out1(pos)');
present2 = find(first_N_ranks2 == net_out2(pos)');
present=(present1+present2)/2;




if isempty(present)
    Accuracy = mean(first_N_ranks)/numel(pos);
    Precision = mean(first_N_ranks)/n;
    MRR =  1/5 * mean(1/rank(1));  % Mean Reciprocal Rank (MRR)
else
    Accuracy = numel(present)/numel(pos);
    Precision = numel(present)/n;
    MRR =  1/5 * mean(1/rank(present(1)));
end

Eval = [Accuracy Precision MRR];

end























