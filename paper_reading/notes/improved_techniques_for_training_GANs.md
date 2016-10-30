# Improved Techniques for Training GANs

--------------------
### Introduction
- The goal of GAN (Generative adversarial networks) 
    - To train a generator network which produces samples from the data distribution by transforming vectors of noise.  
    - Training signal for generator network is provided by a discriminator network 
- Often failed to converge when training GAN
    - Hard to find Nash equilibrium of a non-convex game with continuous, high-dimensional parameters
- Introduce several techniques intended to encourage convergence of the GAN
    - Motivated by a heuristic understanding of the non-convergence probelm 
    - Helps to improve semi-supervised learning performance and improved sample generation
 
### Prior work
- Several papers which have been tried to improve GAN framework
    - DCGAN architecture
    - Feature matching - maximum mean discrepancy to train generator networks
    - Minibatch features - based in part on ideas of batch normalization 

### Toward Convergent GAN training
- Hard to meet the Nash equilibrium
    - E.g., minimizes xy with respect to x and another player minimizes -xy with respect to y
        - Especially SGD with non-convex, high-dimensional data 
- Feature matching
    - Specifying a new objective for the generator 
        - Instead of directly maximizing the output of the discriminator, the new objective requires the generator to generate data that matches the statistics of the real data
            - Train the generator to match the expected value of the features on an intermediate layer of the discriminator
            - ||E_x f(x) - E_z f(G(z))||^2
    - Found that it is useful in situations where regular GAN becomes unstable
- Minibatch discrimination
    - Often generator collapse to a parameter setting where it always emits the same point
    - Minibatch discrimination
        - Let discriminator to look at multiple examples in combination
        - Input x_i - feature f(x_i) - multiply the vector f(x_i) by a tensor T - results in a matrix M_i - compute the L1 distance between rows of matrix Mi 
        - This distance vector is concatenate the output with the intermediate features f(x_i), which are fed into the next layer of the discriminator 
    - Minibatch discriminator allows us to generate visually appealing samples very quickly - superior to feature matching
        - When semi-supervised learning, feature matching was found much better
- Historical averaging
    - Modify the cost of each player to include a term ||\theta - 1/t\Sigma\theta[i]||^2
- One-sided label smoothing
    - Replaces the target value with smoothed values, like .9 or .1 
    - Smooth only the positive labels
- Virtual batch normalization
    - Batch normalization causes the output of the NN to be highly dependent on several other inputs in the same minibatch
    - Each example x is normalized based on the statistics collected on a reference batch of examples that are chosen once and fixed at the start of training
    - Computationally demanding due to two minibatches of data : only use in the generator network
