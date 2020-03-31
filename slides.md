---
marp: true
theme: default
class:
    - lead
    - invert
---


# Introduction to Natural Gradient Descent and Stochastic Reconfiguration

###### Ejaaz Merali

---

# Outline

- Derivation of NGD
- NGD in unsupervised learning
- Derivation of Stochastic Reconfiguration

---

# Some definitions

- $\mathcal{W}$ is our parameter space
- $\mathcal{H}$ is our space of probability functions

- $p_\theta \in \mathcal{H}$ is a parametrized density with parameter vector $\theta \in \mathcal{W}$

- Unsupervised learning involves fitting $p_\theta(x)$ to samples drawn from an unknown distribution $q(x)$

- Supervised learning can be phrased as a procedure in which we find a conditional density $p_\theta(y | x)$ given training samples $(x, y)$

---

# NGD Derivation

- Given a functional $F: \mathcal{H} \to \mathbb{R}$, want to perform an iterative optimization procedure

- We have $p_{\theta_t}$, want the next step, which decreases $F$ further: $p_{\theta_{t+1}}$

- For simplicity, let's drop the $\theta$s for a bit

- Our optimization step can be written as:

$$p_{t+1} = \underset{p}{\operatorname{argmin}} \ F[p]$$

- But where's the $p_t$ dependence?

---


- We want to prevent $p_{t+1}$ from straying too far from $p_t$ in the space of probability distributions

- Do this by selecting some small $\epsilon > 0$, and constrain the KL divergence:

$$D(p_t || p_{t+1}) \leq \epsilon$$

- Our optimization problem is thus:

$$p_{t+1} = \underset{p}{\operatorname{argmin}} \ F[p]
\qquad \text{st.} \quad D(p_t || p) \leq \epsilon
$$

- or, in terms of the parameters, $\theta$:

$$\theta_{t+1} = \underset{\theta}{\operatorname{argmin}} \ F(\theta)
\qquad \text{st.} \quad D(p_{\theta_t} || p_\theta) \leq \epsilon
$$

---

- If we assume that $F$ is well behaved, in the sense that we can Taylor expand it in terms of the parameters, $\theta$, of our model, we have:

$$\theta_{t+1} = \underset{\theta}{\operatorname{argmin}} \ F(\theta_t)
+ \nabla F(\theta_t)^T (\theta - \theta_t)
\qquad \text{st.} \quad D(p_{\theta_t} || p_\theta) \leq \epsilon
$$

- $F(\theta_t)$ is a constant wrt to the optimization, hence:

$$\theta_{t+1} = \underset{\theta}{\operatorname{argmin}} \ \nabla F(\theta_t)^T (\theta - \theta_t)
\qquad \text{st.} \quad D(p_{\theta_t} || p_\theta) \leq \epsilon
$$

- We now need to take a moment to study the KL divergence of two distributions that are close together

---

$$
\begin{aligned}
D(p_{\theta} || p_{\theta + \delta\theta}) &= \mathbb{E}_{p_\theta} \left\lbrack\ln\frac{p_\theta}{p_{\theta + \delta\theta}}\right\rbrack = \mathbb{E}_{p_\theta} \left\lbrack\ln(p_\theta) - \ln(p_{\theta + \delta\theta})\right\rbrack \\
&\approx \mathbb{E}_{p_\theta} \left\lbrack\ln(p_\theta) - \ln(p_{\theta}) - (\partial_i\ln p_\theta) \delta\theta_i - \frac{1}{2} \delta\theta_i (\partial_i\partial_j \ln p_\theta) \delta\theta_j\right\rbrack \\

&= -\mathbb{E}_{p_\theta} \left\lbrack\partial_i\ln p_\theta\right\rbrack\delta\theta_i - \frac{1}{2} \delta\theta_i \mathbb{E}_{p_\theta}\left\lbrack\partial_i\partial_j \ln p_\theta\right\rbrack\delta\theta_j \\

\end{aligned}
$$

Focus on first term:

$$\mathbb{E}_{p_\theta} \left\lbrack\partial_i\ln p_\theta\right\rbrack
= \operatorname{tr}\left\lbrack p_\theta \partial_i\ln p_\theta \right\rbrack
= \operatorname{tr}\left\lbrack \frac{p_\theta}{p_\theta} \partial_i p_\theta \right\rbrack
= \partial_i\operatorname{tr}\left\lbrack p_\theta \right\rbrack
= \partial_i (\text{const})
= 0
$$

---

Second term:

$$
\begin{aligned}
\mathbb{E}_{p_\theta}\left\lbrack\partial_i\partial_j \ln p_\theta\right\rbrack
&= \mathbb{E}_{p_\theta}\left\lbrack\partial_i\frac{\partial_j p_\theta}{p_\theta}\right\rbrack \\
&= \mathbb{E}_{p_\theta}\left\lbrack\frac{\partial_i\partial_j p_\theta}{p_\theta} - \frac{(\partial_i p_\theta)(\partial_j p_\theta)}{p_\theta^2}\right\rbrack \\
&= \mathbb{E}_{p_\theta}\left\lbrack\frac{\partial_i\partial_j p_\theta}{p_\theta}\right\rbrack - \mathbb{E}_{p_\theta}\left\lbrack(\partial_i \ln p_\theta)(\partial_j \ln p_\theta)\right\rbrack \\
&= \operatorname{tr}\left\lbrack p_\theta\frac{\partial_i\partial_j p_\theta}{p_\theta}\right\rbrack - \mathbb{E}_{p_\theta}\left\lbrack(\partial_i \ln p_\theta)(\partial_j \ln p_\theta)\right\rbrack \\
&= \partial_i\partial_j \operatorname{tr}\left\lbrack p_\theta\right\rbrack - \mathbb{E}_{p_\theta}\left\lbrack(\partial_i \ln p_\theta)(\partial_j \ln p_\theta)\right\rbrack \\
&= -\mathbb{E}_{p_\theta}\left\lbrack(\partial_i \ln p_\theta)(\partial_j \ln p_\theta)\right\rbrack \\
\end{aligned}
$$

---

The KL divergence is thus:


$$
\begin{aligned}
D(p_{\theta} || p_{\theta + \delta\theta})
&\approx -\mathbb{E}_{p_\theta} \left\lbrack\partial_i\ln p_\theta\right\rbrack\delta\theta_i - \frac{1}{2} \delta\theta_i \mathbb{E}_{p_\theta}\left\lbrack\partial_i\partial_j \ln p_\theta\right\rbrack\delta\theta_j \\
&= \frac{1}{2} \delta\theta_i \mathbb{E}_{p_\theta}\left\lbrack(\partial_i \ln p_\theta)(\partial_j \ln p_\theta)\right\rbrack \delta\theta_j \\
&= \frac{1}{2} \delta\theta_i g_{ij} \delta\theta_j \\
\end{aligned}
$$

where $g_{ij} = \mathbb{E}_{p_\theta}\left\lbrack(\partial_i \ln p_\theta)(\partial_j \ln p_\theta)\right\rbrack$ is the Fisher Information Metric. We'll call the matrix form $G(\theta)$.

---

Returning to our optimization problem:

$$\theta_{t+1} = \underset{\theta}{\operatorname{argmin}} \ \nabla F(\theta_t)^T (\theta - \theta_t)
\qquad \text{st.} \quad D(p_{\theta_t} || p_\theta) \leq \epsilon
$$

Since we're assuming $\theta$ is close to $\theta_t$, the constraint becomes:

$$
\frac{1}{2} (\theta - \theta_t)^T G(\theta_t) (\theta - \theta_t)
\leq \epsilon
$$

Solve the constrained optimization problem using the Lagrange Multiplier: $\frac{1}{\lambda_t}$

$$
\begin{aligned}
\theta_{t+1} &= \underset{\theta}{\operatorname{argmin}} \left\lbrack \nabla F(\theta_t)^T (\theta - \theta_t) +
\frac{1}{2\lambda_t} (\theta - \theta_t)^T G(\theta_t) (\theta - \theta_t)
\right\rbrack \\
&= \theta_t - \lambda_t G^{-1}(\theta_t) \nabla F(\theta_t)
\end{aligned}
$$

---

# Example: Unsupervised Learning

Given a model, $p_\theta(x)$, and samples drawn from a target distribution $q(x)$, find parameters $\theta \in \mathcal{W}$ which minimize the KL divergence $D(q || p_\theta)$

This is equivalent to minimizing the Negative Log Likelihood:
$$F(p_\theta) = D(q || p_\theta) + H(q)$$

where $H(q)$,the entropy of $q(x)$, is a constant wrt $\theta$

The Hessian of $F$ is thus equal to that of the KL:

$$\nabla^2 F(\theta) = \nabla^2 D(q||p_\theta) =
- \mathbb{E}_q\left\lbrack \partial_i\partial_j \ln p_\theta\right\rbrack
\neq \mathbb{E}_q\left\lbrack (\partial_i \ln p_\theta)(\partial_j \ln p_\theta)\right\rbrack
$$

The Fisher Information Metric, however, is:

$$G(\theta) = -\mathbb{E}_{p_\theta}\left\lbrack \partial_i\partial_j \ln p_\theta\right\rbrack
= \mathbb{E}_{p_\theta}\left\lbrack (\partial_i \ln p_\theta)(\partial_j \ln p_\theta)\right\rbrack$$

---

Two key differences:
- The expectations are taken over different distributions. The target distribution for the Hessian, and the model distribution for the F.I. As training progresses, expect $p_\theta$ to approach $q$, and thus, the Fisher will approximate the Hessian.

- The Fisher information matrix can be expressed in terms of first derivatives. Hessian requires second derivatives.

Problem: despite the F.I. matrix's relative simplicity compared to the Hessian, it is still fairly difficult to compute exactly. There are, however, some analytic expressions for relatively simple cases.

---

In practice, people often just approximate the Fisher from training data:

$$
\tilde{G}(\theta)
= \mathbb{E}_{x \in \mathcal{D}}\left\lbrack (\partial_i \ln p_\theta(x))(\partial_j \ln p_\theta(x))\right\rbrack
$$

This is a bit sketchy, given that this isn't approximating the Fisher (sampling from the wrong distribution), nor is it approximating the Hessian (algebraic form is wrong).

For generative modelling purposes, however, we're able to easily draw samples from our model, allowing us to directly approximate the Fisher.

Still need to deal with the fact that this matrix is HUGE. We'll discuss approximations to the F.I. matrix next time :sunglasses:

---

# The Quantum Case

We have a variational trial state $\vert\psi_\theta\rangle$ with $p$ parameters.

Nudging the parameters $\theta \to \theta + \delta\theta$:

$$
\psi\prime(x; \theta) = \psi(x; \theta) + \sum_{k=1}^p \delta\theta_k \frac{\partial}{\partial\theta_k}\psi(x; \theta)
$$

Let $O_k\vert x \rangle = \left(\frac{\partial}{\partial\theta_k}\ln\psi(x; \theta)\right)\vert x \rangle$, $O_0 = 1$, $\delta\theta_0 = 1$

Then $\vert \psi\prime \rangle = \sum_{k=0}^p \delta\theta_k O_k \vert \psi\rangle$

Let $\delta\theta_0$ vary freely, remembering to normalize parameter shifts later: $\delta\theta_k \to \delta\theta_k/\delta\theta_0$

---

Let $P$ be a projection operator st
$$PO_k\vert\psi\rangle = O_k\vert\psi\rangle \quad \forall k = 0,1,\ldots,p$$

Hence $\vert\psi\prime\rangle = P(\Lambda - H)\vert\psi\rangle$

where $\Lambda$ is selected large enough that $\langle \psi\prime\vert H \vert \psi\prime \rangle$ < $\langle \psi\vert H \vert \psi \rangle$

Take overlap wrt states $\langle \psi \vert O_k$

$$\langle \psi \vert O_k \vert\psi\prime\rangle = \langle \psi \vert O_k  P(\Lambda - H)\vert\psi\rangle$$

Expand $\vert\psi\prime\rangle$

$$
\begin{aligned}
\sum_{l=0}^p \delta\theta_l\langle \psi \vert O_k O_l \vert\psi\rangle &= \langle \psi \vert O_k  P(\Lambda - H)\vert\psi\rangle \\
\sum_{l=0}^p \delta\theta_l s_{kl} &= f_k
\end{aligned}
$$

---

We typically use MC sampling from the trial state to determine $s_{kl}, f_k$.

Note, $f_0 = \delta\theta_0 + \sum_{l=1}^p \delta\theta_l s_{kl}$

$\delta\theta_0$ has a direct dependence on $\Lambda$, which affects parameter convergence rate. To fix $\Lambda$ we employ a trick similar to what we did for NGD. Penalize excessive changes to the state:

$$\Delta \psi = \vert \psi\prime - \psi \vert^2 = \sum_{l,k > 0} \delta\theta_k \overline{s_{kl}} \delta\theta_l$$

where $\overline{s_{kl}} = s_{kl} - s_{0k}s_{0l}, \forall l,k > 0$

Introduce a Lagrange Multiplier $\mu$, our new objective function is: $E + \mu \Delta\psi$. The solution becomes:

$$\delta\theta_k = \frac{1}{\mu}\sum_{l=1}^p (\overline{s}^{-1})_{kl} \overline{f}_l \qquad\text{where:}\quad \overline{f}_l = -\frac{\partial E}{\partial \theta_l}$$

---

# Example: Positive Wavefunction:

$$\vert \psi \rangle = \sum_x \sqrt{p(x)}\vert x\rangle$$

Matrix $s$ becomes:

$$
\begin{aligned}
s &= \langle (\partial_i \ln\psi) (\partial_j \ln\psi) \rangle - \langle (\partial_i \ln\psi)\rangle\langle(\partial_j \ln\psi) \rangle \\
&= \frac{1}{4}\mathbb{E}_p\left\lbrack (\partial_i \ln p) (\partial_j \ln p) \right\rbrack
-
\frac{1}{4}\mathbb{E}_p\left\lbrack (\partial_i \ln p) \right\rbrack \mathbb{E}_p\left\lbrack(\partial_j \ln p) \right\rbrack \\
&= \frac{1}{4}\mathbb{E}_p\left\lbrack (\partial_i \ln p) (\partial_j \ln p) \right\rbrack - 0\\
&= \frac{1}{4}G
\end{aligned}
$$

---

# Summary

- We derived NGD using the method of Lagrange Multipliers

- We considered the case of unsupervised learning and compared NGD to the standard 2nd-order optimization procedure

- We analysed the emergence of a similar technique for the optimization of quantum wavefunctions, and showed it to be equivalent to NGD for the case of a positive wavefunction

---

# Next Time:

We'll look at approximations to the Fisher Information Matrix, specifically, the Kronecker Factorization (KFAC) and the Eigenvalue-corrected Kronecker Factorization (EKFAC).

---

# Thank you

---

# References

Amari, Shun-Ichi. "Natural gradient works efficiently in learning." Neural computation 10.2 (1998): 251-276.

Bottou, L., F. E. Curtis, and J. Nocedal. "Optimization methods for large-scale machine learning. arXiv 2016." arXiv preprint arXiv:1606.04838.

Hoffman, Matthew D., et al. "Stochastic variational inference." The Journal of Machine Learning Research 14.1 (2013): 1303-1347.

Park, Chae-Yeun, and Michael J. Kastoryano. "On the geometry of learning neural quantum states." arXiv preprint arXiv:1910.11163 (2019).

Sorella, Sandro, Giuseppe E. Santoro, and Federico Becca. "SISSA Lecture notes on Numerical methods for strongly correlated electrons." (2010).
