function [y] = adaline_Func(X,W,O)
    for a=1:length(X)
                    u = sum((W.*X(a,:)) + O,'all');
                    %% Função de Ativação
                    if u >= 0
                        y(a) = 1;
                    else
                        y(a) = -1;
                    end
    end
    return
end 