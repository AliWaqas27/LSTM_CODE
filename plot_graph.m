function[] = plot_graph()
clear all;
close all





load Eval_all
Eval2 = Eval_all;
Terms = ["Accuracy","MAP","MRR"];
l = ["Dataset1","Dataset2","Dataset3","Dataset4","Dataset5"];
pn = [5: 5: 20];
for i = 1 : 5  %% for all datasets
    for k = 1 : 3  %% for all terms
        for p = 1 : 4 %% for all retrievals
            for j = 1 : 5 %% for all algorithms
                val(p,j) = Eval2{i,p,j}(k);
            end
        end
        val = sort(val);
        figure
        plot(pn,val(:, 1),'-^b', 'LineWidth', 2, 'markersize',10); hold on
        plot(pn,val(:, 2),'-^g', 'LineWidth', 2, 'markersize',10)
        plot(pn,val(:, 3),'-^c', 'LineWidth', 2, 'markersize',10)
        plot(pn,val(:, 4),'-^m', 'LineWidth', 2, 'markersize',10)
        plot(pn,val(:, 5),'-^k', 'LineWidth', 2, 'markersize',10)
        xlabel('No of files Retrieved');
        ylabel(Terms{k});
        h = legend('HGW-SFO-CDNN [27]','GWO-IDN-LSTM [28]','SFO-IDN-LSTM [29]', 'COA-IDN-LSTM [26]', 'MSP-COA-IDN-LSTM');
        set(h,'fontsize',10,'Location','NorthEastOutside')
        print('-dtiff', '-r300', ['.\Results\Alg', Terms{k},l{i}])
        
    end
    
    for k = 1 : 3  %% for all terms
        for p = 1 : 4 %% for all retrievals
            for j = 6 : 10 %% for all Methods
                if j == 10
                    val2(p,5) = Eval2{i,p,5}(k);
                else
                    val2(p,j-5) = Eval2{i,p,j}(k);
                end
            end
        end
        val2 = sort(val2);
        figure
        bar(val2,'Linewidth',2)
        set(gca,'FontSize',14)
        h1 = legend('CNN [3]','HGW-SFO-CDNN [27]','LSTM [7]','CNN-DNN-LSTM [30]','MSP-COA-IDN-LSTM');
        set(h1,'FontSize',12,'Location','NorthEastOutside');
        ylabel(Terms{k},'FontSize',14);
        xlabel('No of files Retrieved','FontSize',14);
        xticklabels({'5','10','15','20'})
        print('-dtiff', '-r500', ['.\Results\Models', Terms{k},l{i}]);
        





    end
end

end