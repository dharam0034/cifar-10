function [mu sigma p]= cifar_10_bayes_learn(tr_f,tr_l)
    minl=min(tr_l);maxl=max(tr_l);
    
    mu={};
    sigma={};
    p=[];
    for i=minl:maxl
        inds = find(tr_l==i);
        p=[p; length(inds)/length(tr_l)];
        mu{i+1}=[mean(tr_f(inds,1)) mean(tr_f(inds,2)) mean(tr_f(inds,3))];
        sigma{i+1}=[std(tr_f(inds,1)) std(tr_f(inds,2)) std(tr_f(inds,3))];
    end
end