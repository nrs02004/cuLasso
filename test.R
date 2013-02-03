softthreshold = function(y, lambda) {
  sign(y) * (abs(y) - lambda) * sign(abs(y) > lambda)
}

mylasso = function(X, y, lambda, t = 1e-4, max.iters = 100) {
  intercept = mean(y)
  y = y - mean(y)
  Bnew = Bold = matrix(0, ncol(X), length(lambda))
  XtX = crossprod(X)
  Xty = crossprod(X,y)
  for (i in 1:length(lambda))
  for (iter in 1:max.iters) {
    Bnew[,i] = softthreshold(Bold[,i] + t*(Xty - XtX%*%Bold[,i]), lambda[i])
    if (all(abs(Bnew[,i] - Bold[,i]) < 1e-4)) break
    else Bold[,i] = Bnew[,i]
  }
  return (rbind(intercept, Bnew))
}
