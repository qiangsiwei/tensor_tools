function [S,X,Y,Z] = catd1(A, B, C, D, epsilon, lambda1, lambda2, lambda3, lambda4) 


% size of core Tensor
dimX = 10;
dimY = 10;

% step size
t0 = 10;
t = t0;

dim1 = size(A,1);
dim2 = size(A,2);
dim3 = size(A,3);

%initialize S R U T F G with small random values
X = rand(dim1,dimX) ./ 10
Y = rand(dim2,dimY) ./ 10
Z = rand(dim3,dimX) ./ 10
S = tenrand(dimX,dimY,dimX) ./ 10
U = rand(dimX, size(B, 2)) ./ 10

[indexs,values] = find(A);
turn = 1:length(values);

% initialize function loss
c = zeros(size(values));
for j = 1:length(values)
    ijk = ttv(S,{X(indexs(j,1),:)',Y(indexs(j,2),:)',Z(indexs(j,3),:)'});
    c(j) = ijk;
end
loss_t1 = norm(c - values);
loss_t = loss_t1 + epsilon + 1;
fprintf('loss: %f\n', loss_t1);

% get the laplacian matrices
LC = diag(sum(C)) - C;

while loss_t - loss_t1 > epsilon   
    
    oldX = X;
    oldY = Y;
    oldZ = Z;
    oldS = S;
    
    % optimize each element in randomized sequence   
    for num = 1:length(values)-1     
        change = randi([num+1,length(values)]);
        temp = turn(num);
        turn(num) = turn(change);
        turn(change) = temp;
    end
   
    for num = 1:length(values) % for every nonzero entries in A
        if (isnan(S(1,1,1)))  % check for NAN
            disp nanerror;
            return;
        end
        
        tnum = turn(num);
        nita = 1/sqrt(t);  % step size
        t = t + 1;
        i = indexs(tnum,1);
        j = indexs(tnum,2);
        k = indexs(tnum,3);
        
        Xi = X(i,:)';
        Yj = Y(j,:)';
        Zk = Z(k,:)';
               
      
        Fijk = double(ttensor(S,{Xi',Yj',Zk'}));
        
        Yijk = values(tnum);
        Lfy = Fijk - Yijk;
        
        nita
        XLfy = double((nita * Lfy) * ttv(S,{Yj,Zk},[2 3]));
        YLfy = double((nita * Lfy) * ttv(S,{Xi,Zk},[1 3]));
        ZLfy = double((nita * Lfy) * ttv(S,{Xi,Yj},[1 2]));
        SLfy = (nita * Lfy) * tensor(ktensor({Xi,Yj,Zk}));
        
        
        X(i,:) = ((1 - nita * lambda4) * Xi - XLfy)' - lambda1 * (nita * (Xi' * U - B(i, :)) * U') - lambda3 * (nita * (Xi' * Z' - D(i, :)) * Z);
        LCY = nita * LC * Y;
        Y(j,:) = ((1 - nita * lambda4) * Yj - YLfy)' - lambda2 * LCY(j,:);       
        Z(k,:) = ((1 - nita * lambda4) * Zk - ZLfy)' - lambda3 * (nita * (Zk' * X' - D(:, k)') * X);
        S = (1 - nita * lambda4) * S - SLfy;
                
        U = U - lambda1 * nita * ((Xi' * U - B(i, :))' * Xi')' - nita * lambda4 * U;
        
    end
    
    % compute function loss 
    c = zeros(size(values));
    for j = 1:length(values)
        ijk = ttv(S,{X(indexs(j,1),:)',Y(indexs(j,2),:)',Z(indexs(j,3),:)'});
        c(j) = ijk;
    end
    loss_t = loss_t1;
    loss_t1 = norm(c - values);
    fprintf('loss: %f\n', loss_t1);    
end
fprintf('end\n');    

X = oldX;
Y = oldY;
Z = oldZ;
S = oldS;

end
