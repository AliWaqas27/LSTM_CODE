function[] = plot_results()
clc;
clear all;
close all;
load Eval_all
Eval = Eval_all;


disp('%%%%%%%%%%%%%%%%%%   Algorithm_Analysis   %%%%%%%%%%%%%%%%%%%%%%');
for i = 1 : 5 %% for all datasets
    disp(strcat('****************** ',' Dataset','-',num2str(i),' ******************'));
    ln = {'Accuracy','MAP','MRR'};
    T = table(Eval{i,4,1}(1:end),Eval{i,4,2}(1:end),Eval{i,4,3}(1:end),Eval{i,4,4}(1:end),Eval{i,4,5}(1:end),'Rownames',ln);
    T.Properties.VariableNames = {'GWO-SFO','GWO','SFO','COA', 'Proposed'};
    disp(T)
end

disp('%%%%%%%%%%%%%%%%%%   Classifier_Analysis   %%%%%%%%%%%%%%%%%%%%%%');
for i = 1 : 5 %% for all datasets
    disp(strcat('****************** ',' Dataset','-',num2str(i),' ******************'));
    ln = {'Accuracy','MAP','MRR'};
    T = table(Eval{i,4,6}(1:end),Eval{i,4,7}(1:end),Eval{i,4,8}(1:end),Eval{i,4,9}(1:end),Eval{i,4,5}(1:end),'Rownames',ln);
    T.Properties.VariableNames = {'CNN','CNN-DNN','LSTM','CNN+DNN+LSTM','Proposed'};
    disp(T)
end


end