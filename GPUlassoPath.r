activeGPUlassoPath <- function(X, y, beta = matrix(0,ncol = length(lambda), nrow = ncol(X)), maxIt = 1000, thresh = 0.000001, step_size= 0.1, lambda = 1, GPU = 1){
  if(GPU == 1){
    if(!is.loaded("activePathSol")){
      dyn.load("~/software/CUDAstuff/RgradSolve/finalPath/GPUlassoPath.so")
    }
  }
  
  n <- nrow(X)
  p <- ncol(X)

  
  
  if(GPU == 1){
    fit <- .C("activePathSol", X = as.single(X), y = as.single(y), n = as.integer(n), p = as.integer(p), maxIt = as.integer(maxIt), thresh = as.single(thresh), step_size= as.single(step_size), lambda = as.single(lambda), beta = as.single(beta), num_lambda = as.integer(length(lambda)))
  }
fit$beta <- matrix(fit$beta, nrow = p, byrow = F)
  return(fit)
}
