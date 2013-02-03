#' Lasso on a GPU
#'
#' Entry point to CUDA implementation of lasso
#' @param X design matrix X
#' @param y response vector y
#' @param beta initial value for beta matrix for varying lambda penalty
#' @param maxIt maximum iterations
#' @param thresh convergence threshold
#' @param step_size step size for gradient descent
#' @param lambda l1 penalties
#' @param GPU if true, run on GPU using CUDA
#' @useDynLib GPULassoPath
#' @export
activeGPULassoPath <- function(X, y, beta = matrix(0,ncol = length(lambda), nrow = ncol(X)),
                               maxIt = 1e3, thresh = 1e-6, step_size= 0.1, lambda = 1,
                               GPU = TRUE) {
  
  n <- nrow(X)
  p <- ncol(X)

  if (GPU) {
    fit <- .C("activePathSol", X = as.single(X), y = as.single(y), n = as.integer(n),
              p = as.integer(p), maxIt = as.integer(maxIt), thresh = as.single(thresh),
              step_size= as.single(step_size), lambda = as.single(lambda),
              beta = as.single(beta), num_lambda = as.integer(length(lambda)))
  }
  fit$beta <- matrix(fit$beta, nrow = p, byrow = F)
  return (fit)
}
