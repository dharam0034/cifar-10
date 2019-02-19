function [CA]=cifar_10_evaluate(pred,gt)
     CA=(numel(find(pred==gt))/10000)*100;
end