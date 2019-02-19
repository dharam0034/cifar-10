function [NN]=cifar_10_1NN(x,tr_data,tr_labels)
    euclidean_distance=[];
    tm = repmat(x,length(tr_data),1);
    euclidean_distance=sum(abs(minus(double(tr_data),double(tm))),2);
    [M,I] = min(euclidean_distance);
    NN=tr_labels(I);
end