SuperLearner <- function(y, experts, awake = NULL, loss.type = "square", 
                w0 = NULL, training = NULL, quiet = FALSE) {
  experts <- as.matrix(experts)
  
  N <- ncol(experts)  # Number of experts
  T <- nrow(experts)  # Number of instants
  
  pred <- rep(0, T)  # Prediction vector
  weights <- matrix(0, ncol = N, nrow = T)  # Matrix of weights formed by the mixture

  equal_weights = rep(1/N, N)
  eqfun <- function(alphas) sum(alphas)
  
  if (! quiet) steps <- init_progress(T)
  
  cumulativeLoss <- 0
  for (t in 1:T) {
    if (! quiet) update_progress(t, steps)

    risk <- function(weights) {
      weights <- weights / sum(weights)
      losses <- numeric(t - 1)
      for(i in 1:(t - 1)) {
        pred <- experts[i, ] %*% weights
        losses[i] <- loss(pred, y[i], pred, loss.type, loss.gradient = FALSE)
      }
      mean(losses)
    }
    
    # Find the weights which, in hindsight, would have minimized the loss
    if(t == 1) {
      weights[t, ] <- equal_weights
    }
    else {
      #fit <- solnp(equal_weights, risk, eqfun = eqfun, eqB = 1, LB = rep(0, N))
      tryCatch({
        fit <- optim(equal_weights, risk, lower = 0, method = "L-BFGS-B", control = list(trace = 1))
      }, error = function(e) {
        print(e)
        fit <- list(par = weights[t - 1, ])
      })
      
      # Weight update
      weights[t,] <- fit$par
      weights[t, ] <- weights[t, ] / sum(weights[t,])
      
      #if(fit$convergence != 0) {
        #weights[t, ] <- weights[t - 1, ]
        #browser()
      #}
    }
    
    #if(t == 300) browser()
    print(paste0(t, "/", T))
    
    # Prediction and losses
    pred[t] <- experts[t, ] %*% weights[t, ]
    
    cumulativeLoss <- cumulativeLoss + loss(x = pred[t], y = y[t], loss.type = loss.type, loss.gradient = FALSE)
  }
  if (! quiet) end_progress()

  w = weights[nrow(weights),]
  
  object <- list(model = "SuperLearner", loss.type = loss.type, loss.gradient = FALSE,
                 coefficients = w)
  
  #R <- R.w0 - log(w0) / eta
  object$parameters <- list()
  object$weights <- weights
  object$prediction <- pred
  
  #object$training <- list(R = R, w0 = w0, cumulativeLoss = cumulativeLoss)
  object$training <- list(cumulativeLoss = cumulativeLoss)
  
  return(object)
} 
