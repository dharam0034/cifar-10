function [BC] = cifar_10_bayes_classify_1(f,mu,Sigma,p)
    
%     f 
%     mu{1}
%     Sigma{1}
%     mvnpdf(f,mu{1},Sigma{1})

    for i=1:10
        posteriori(i)= mvnpdf(f,mu{i},Sigma{i})*p{i};
    end
    BC=find(posteriori == max(posteriori(:)))-1;
end

