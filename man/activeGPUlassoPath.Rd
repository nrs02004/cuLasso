\name{activeGPULassoPath}
\alias{activeGPULassoPath}
\title{Lasso on a GPU}
\usage{
  activeGPULassoPath(X, y,
    beta = matrix(0, ncol = length(lambda), nrow = ncol(X)),
    maxIt = 1000, thresh = 1e-06, step_size = 0.1,
    lambda = 1, GPU = TRUE)
}
\arguments{
  \item{X}{design matrix X}

  \item{y}{response vector y}

  \item{beta}{initial value for beta matrix for varying
  lambda penalty}

  \item{maxIt}{maximum iterations}

  \item{thresh}{convergence threshold}

  \item{step_size}{step size for gradient descent}

  \item{lambda}{l1 penalties}

  \item{GPU}{if true, run on GPU using CUDA}
}
\description{
  Entry point to CUDA implementation of lasso
}

