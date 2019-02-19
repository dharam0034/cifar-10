function [BC] = cifar_10_bayes_classifyEx(f,mu_Ex,Sigma_Ex,p_Ex)
    
%     f 
%     mu{1}
%     Sigma{1}
%     mvnpdf(f,mu{1},Sigma{1})

    for i=1:10
        posteriori(i)= mvnpdf(f,mu_Ex{i},Sigma_Ex{i})*p_Ex{i};
    end
    BC=find(posteriori == max(posteriori(:)))-1;
end

