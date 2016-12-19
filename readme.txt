RNACompare Executable Code for Academic Users, Version 1.0
=======================================================

-------------
Introduction
-------------

RNACompare is a tool package to address RNA secondary structure comparison and classification issues thoroughly. 
It includes two sub directories:
(1). RNASim: This is a C executable software used for RNA comparsion based on grammar inferense.
(2). IndefiniteSVM: This is a matlab script package used for RNA classification based on indefinite kernel learning.



-------------
Platform
-------------

Windows 2000/XP.

-------------
Usage
-------------

(1).RNASim

RNASim File1 File2 (eg. RNASim tRNA.txt tRNA.txt > output )

File1 and File2 are two input files with a set of RNA primary sequences with their corresponding secondary structures.
Output of this comparison is a deference matrix. Row i,column j in the matrix correpondings to the quantitive distance([0,1]) of RNA i in File1 compared with RNA j in File2.

Note: If File1 and File2 are identical, then the output matrix is the pair-wise difference of all the RNA secondary structures in one file. 


    
(2).IndefiniteSVM

example script in matlab:

addpath('D:\MATLAB701\work\libsvm-mat-2.9-1');
addpath(genpath(pwd)); % add to path current directory and subdirectories mexFunctions and subRoutines    
% parameters
acc=.01;
maxiters=2000;
info=100;
C=10;
rho=10;
stepsize=5; % only for Projected Gradient Algorithm

  p1=ones(20,1);n1=-1*ones(20,1);a1=[p1;n1];
  p2=ones(20,1);n2=-1*ones(20,1);a2=[p2;n2];
  p3=ones(20,1);n3=-1*ones(20,1);a3=[p3;n3];
  p4=ones(20,1);n4=-1*ones(20,1);a4=[p4;n4];
  p5=ones(20,1);n5=-1*ones(20,1);a5=[p5;n5]; 
  % This is for 5-fold cross validation
  
  labelsTrain=[a1;a2;a3;a4];labelsTest=a5;%4/5 data for training and 1/5 data for testing
  numTrain=length(labelsTrain);
  K=dlmread('miRNA_distance_matrix.txt'); %an 200*200 distance matrix for miRNA, this matrix is obtained by RNASim on the comparsion of RNAs from miRNA.txt
  K=1-K; %transfer distancematrix to similarity matrix
  
% Train using Projected Gradient Algorithm
[alpha,primal,dual,gap]=IndefiniteSVM(K(1:numTrain,1:numTrain),labelsTrain,C,rho,acc,info,maxiters,stepsize,1);
% Test using Results of Projected Gradient Algorithm
[pred,accuracy,sensitivity,specificity]=IndSVMerror(alpha,labelsTrain,labelsTest,K,C,rho); % get the error
    
    
    
    
    
    
  Lihua Tang   
   
  Tanglihua2002@163.com   
  
  ============================================================
  
  Last updated on Dec.19, 2016"
       
    
      

