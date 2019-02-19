function [mu Sigma p]= cifar_10_bayes_learn_1(tr_f,tr_l)
    minl=min(tr_l);maxl=max(tr_l);
    
    mu={};
    Sigma={};
    p={};
    for i=minl:maxl
        inds = find(tr_l==i);
        p{i+1}=[length(inds)/length(tr_l)];
        mu{i+1}= [mean(tr_f(inds,1)) mean(tr_f(inds,2)) mean(tr_f(inds,3))];
        RGB=[tr_f(inds,1) tr_f(inds,2) tr_f(inds,3)];
        Sigma{i+1}=cov(RGB);
    end
end