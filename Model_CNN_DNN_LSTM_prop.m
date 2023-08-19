function[Eval] = Model_CNN_DNN_LSTM_prop(Feat,All_Scores, sol, n, pos)


fetaureWeight = double(Feat) .* repmat(double(sol(1:16)),length(Feat),1);


% Deep Neural Network (DNN)
net = feedforwardnet([round(sol(17)) round(sol(18))]);
Feat = double(fetaureWeight);
Tar = All_Scores;
net_1 = train(net,Feat',Tar);               % train data
net_out1 = net_1(Feat');
rank1 = sort(net_out1, 'descend');
first_N_ranks1 = rank1(1:n);
% LSTM
net_out2 = Model_LSTM(Feat',Tar, Feat', round(sol(19)));
rank2 = sort(net_out2, 'descend');
rank = (rank1+rank2)/2;
first_N_ranks2 = rank2(1:n);
present1 = find(first_N_ranks1 == net_out1(pos)');
present2 = find(first_N_ranks2 == net_out2(pos)');
presesnt=(present1+present2)/2;
if isempty(presesnt)
    Accuracy = mean(first_N_ranks)/numel(pos);
    Precision = mean(first_N_ranks)/n;
    MRR =  1/5 * mean(1/rank(1));  % Mean Reciprocal Rank (MRR)
else
    Accuracy = numel(presesnt)/numel(pos);
    Precision = numel(presesnt)/n;
    MRR =  1/5 * mean(1/rank(presesnt(1)));
end

Eval = [Accuracy Precision MRR];

end























