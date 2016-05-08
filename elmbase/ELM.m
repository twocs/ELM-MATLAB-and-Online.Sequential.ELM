function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy, TY] = ELM(TrainingData, TestingData, varargin)

% Usage: elm(TrainingData, TestingData)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, 'Elm_Type', 'sig', 'NumberofHiddenNeurons', 20, 'ActivationFunction', 'hardlim', 'C', 1, 'display', false)
%
% Input:
% TrainingData     - training data struct, with TrainingData.P being inputs, TrainingData.T being labels
% TestingData      - testing data struct, with TestingData.P being inputs, TestingData.T being labels

% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TY                          - Predictions for TestingData
%%
    %%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       APRIL 2004
	
	%%% revisions to input from variables, not files, by Tom Anderson, 2016
	
p = inputParser;
p.addRequired('TrainingData', @isstruct);
p.addRequired('TestingData', @isstruct);

p.addParameter('C', 1, @isdouble); % scaling factor
p.addParameter('Elm_Type', 1, @isdouble); % type of Elm
p.addParameter('NumberofHiddenNeurons', 20, @isdouble);
p.addParameter('ActivationFunction', 'sig', @ischar);
p.addParameter('display', true, @islogical); % display to console
p.addParameter('RegularizationMethod', 'standard', @ischar); % display to console
p.parse(TrainingData, TestingData, varargin{:});
pp = p.Results;

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;


TV.T = TestingData.T;
TV.P = TestingData.P;
T = TrainingData.T;
P = TrainingData.P;

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(pp.NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(pp.NumberofHiddenNeurons,1);
tempH=InputWeight*P;
% clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch pp.ActivationFunction
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here
	otherwise % @ta pass in activation functions as strings
		H = eval(sprintf('%s(tempH)', ActivationFunction));
end
% clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
switch pp.RegularizationMethod
    case 'standard'
        OutputWeight=pinv(H') * T';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper
    case 'method1'
        OutputWeight=inv(eye(size(H,1))/C+H * H') * H * T';   % faster method 1 //refer to 2012 IEEE TSMC-B paper
    case 'method2'
        %implementation; one can set regularizaiton factor C properly in classification applications 
        OutputWeight=(eye(size(H,1))/C+H * H') \ H * T';      % faster method 2 //refer to 2012 IEEE TSMC-B paper
        %implementation; one can set regularizaiton factor C properly in classification applications
        %If you use faster methods or kernel method, PLEASE CITE in your paper properly: 
        %Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang, "Extreme Learning Machine for Regression and Multi-Class Classification," submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence, October 2010. 
end

end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;        %   Calculate CPU time (seconds) spent for training ELM
if pp.display 
    eval('TrainingTime'); 
end
%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
if pp.Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y));           %   Calculate training accuracy (RMSE) for regression case
    if pp.display 
        eval('TrainingAccuracy'); 
    end
end
% clear H;

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
% clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch pp.ActivationFunction
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
		%%%%%%%% More activation functions can be added here        
	otherwise % @ta pass in activation functions as strings
		H_test = eval(sprintf('%s(tempH_test)', ActivationFunction));

end
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
if pp.display 
    eval('TestingTime'); 
end
    
if pp.Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY));            %   Calculate testing accuracy (RMSE) for regression case
    if pp.display 
        eval('TestingAccuracy'); 
    end
end

if pp.Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;

    for i = 1 : size(T, 2)
        [~, label_index_expected]=max(T(:,i));
        [~, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);
    if pp.display 
        eval('TrainingAccuracy'); 
    end
    for i = 1 : size(TV.T, 2)
        [~, label_index_expected]=max(TV.T(:,i));
        [~, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);
    if pp.display 
        eval('TestingAccuracy'); 
    end    
end
