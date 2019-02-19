function [BC] = cifar_10_bayes_classify(f,mu,sigma,p)
    for i=1:10
       posteriori(i)= normpdf(f(1),mu{i}(1),sigma{i}(1))*normpdf(f(2),mu{i}(2),sigma{i}(2))*normpdf(f(3),mu{i}(3),sigma{i}(3))*p(i);
    end
    BC=find(posteriori == max(posteriori(:)))-1;
end