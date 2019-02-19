function [mu_Ex, Sigma_Ex, p_Ex]= cifar_10_bayes_learnEx(tr_f,tr_l)
    minl=min(tr_l);maxl=max(tr_l);
    mu_Ex={};
    Sigma_Ex={};
    p_Ex={};
    
    for i=minl:maxl
        inds = find(tr_l==i);
        p_Ex{i+1}=[length(inds)/length(tr_l)];
        mu_Ex{i+1}= [mean(tr_f(inds,:))];
        RGB=[tr_f(inds,:)];
        Sigma_Ex{i+1}=cov(RGB);
    end
end