##LPSVM - MSc Project

A version of the thesis can be found in [LPSVM 1.1.pdf](https://github.com/ornithos/lpsvm/blob/master/LPSVM%20-%20v1.1.pdf "LPSVM 1.1"). The algorithm is based on LPBoost, used to optimise the SVM objective, a method proposed by John Shawe-Taylor as a possible means of optimising kernel SVM in a distributed setting. The framework is attractive in that it describes a path of support vector sets that is guaranteed to converge to the SVM solution, but when stopped early will usually have far fewer support vectors. Due to the boosting strategy, the performance is strong throughout the pathway.

This project contains all of the code used to prototype the LPSVM algorithm. The experiments may be replicated easily once the relevant datasets have been downloaded. The 'Optim Ideas' folder contains the various investigations into alternative and distributed linear programming. Further work using a stochastic subgradient approach is available from David Martinez-Rego on request. Auxiliary scripts used for testing and cross validation are also available in the relevant folder.

More details of the algorithm are given in the thesis.

### Code structure

The main calling function is lpsvm. Dependencies are shown below, with returning arrows indicate the top level control flow.

                +------------------------+           +--------------------+           
                v                        +           v                    +           
            lpsvm +--------------------> solveLPActive +----------------> solveWeights
              +                           +                                           
              +----> initModels           +----> chooseInitCol                        
              |----> initModelMatrix      |----> (pickIndependentCols)                
              |----> solveModel           +----> createPrimalProblem                  
              |----> newModelRow                                                      
              |----> countSV                                                          
              +----> kkMat                                                            
                                                                                      
                                                                                      
                kkMat +-->}                                                               
                          }kernelCache                                                    
    @kernel  +----------->}                                                               
                                                                                      
                                                                                      
            predictLPSVM                                                              
              +                                                                       
              +---->  chunkVector                                                     
              +---->  kkMat

The functions lpsvmMNIST and lpsvmTOY2 are examples of using lpsvm on the MNIST and a toy gaussian dataset respectively.
