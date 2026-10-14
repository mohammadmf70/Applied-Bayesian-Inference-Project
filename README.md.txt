# Applied Bayesian Inference Project: From Belief to Computation

<p align="center">
  <b>A Comprehensive 10-Week Journey into Bayesian Statistics, Probabilistic Modeling, and Machine Learning</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Course-Probability%20&%20Statistics-blue.svg" alt="Course">
  <img src="https://img.shields.io/badge/Institution-Sharif%20University%20of%20Technology-orange.svg" alt="Sharif University">
  <img src="https://img.shields.io/badge/Python-3.x-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/Status-Completed-success.svg" alt="Status">
</p>

---

## 👥 Project Team
* **Mohammad Mahdi Fotuhi** (Sharif University of Technology)
* **Mohammad Mahdi Barzkar**
* **Mohammad Yasin Karimi**
* **Hadi Zehtab**

---

## 📖 Overview
Welcome to the repository for our **Applied Bayesian Inference Project**, developed as part of the Probability and Statistics course. This project bridges the gap between classical (Frequentist) statistics—where parameters are treated as fixed, unknown constants—and Bayesian statistics, where parameters are treated as random variables described by probability distributions.

Rather than relying solely on black-box libraries, this project focuses on **first-principles implementation** using Python (NumPy, SciPy, Matplotlib) to build foundational Bayesian machinery from scratch, ranging from analytical coin-tossing updates to Bayesian Linear Regression, A/B Testing decision theory, and custom Naive Bayes classifiers.

---

## 🗂️ Project Structure & Modules

### Part I: Foundations of Belief (Discrete & Initial Continuous Models)
* **Week 1: The Mechanics of Uncertainty**
  * Conditional probability as "universe shrinking" and formalizing Bayes' Theorem.
  * Solving classical paradoxes (The Medical Test Paradox, The Three-Card Swindle, The Monty Hall Problem).
  * **Simulation:** Simulating a Bayesian agent learning a secret coin bias ($p=0.7$) over 100 flips using the Beta-Binomial conjugate update.
* **Week 2: Updating Beliefs & Bayesian Linear Regression (Grid Approximation)**
  * Discrete hypothesis updates using the *Bayes Table Method* (The Cookie Problem, Three-Coin Puzzle, and Sherlock Holmes Die Falsification principle).
  * **Machine Learning Core:** Implementing Bayesian Linear Regression via **Grid Approximation** ($50 \times 50$ parameter space for slope $m$ and intercept $c$). Maximizing Gaussian likelihood, computing posterior grids, tracking entropy/MAP, and sampling regression lines from the joint posterior.

### Part II: Continuous Probability & Decision Making
* **Week 3: The Beta Distribution & Ranking Systems**
  * Exploring conjugate priors and pseudo-counts ($\alpha, \beta$). Connecting Laplace Smoothing directly to Bayesian expected values under a $\text{Beta}(1,1)$ prior.
  * **Application:** Building a robust movie rating and recommendation system (ranking items via the **5th Percentile Lower Bound** of the posterior distribution to penalize uncertainty, solving the Frequentist failure on low-sample items).
* **Week 4: Quantifying Risk & Credible Intervals**
  * Contrast between Frequentist Confidence Intervals and Bayesian Credible Intervals.
  * Exploring Equal-Tailed Intervals (ETI) vs. Highest Density Intervals (HDI) and handling bimodal distributions.
  * **Simulation:** The Overlap Fallacy in A/B testing and Monte Carlo estimation of the *Probability of Superiority*.

### Part III: Bayesian Decision Making & Model Checking
* **Week 5: Bayesian A/B Testing & Decision Theory**
  * Moving beyond simple significance to **Effect Size (Delta Method)** and **Expected Loss** calculation.
  * Decision-making under utility functions and economic constraints (e.g., assessing whether a conversion rate lift justifies a high deployment cost).
* **Week 6: Model Checking (Posterior Predictive Checks)**
  * Generative modeling and Turing tests for statistical models.
  * **The Euro Problem:** Simulating 250 spins of a Belgian Euro coin (140 heads) to test the Skeptic's "Fair Coin" hypothesis versus the Learned Bayesian model using Posterior Predictive Checks (PPC).

### Part IV: Bayesian Classification & Evaluation
* **Week 7: Naive Bayes Classifier from Scratch**
  * Bag-of-Words text representation and the "Naive" conditional independence assumption.
  * Solving the **Underflow Problem** using log-probabilities ($\log \sum$) and addressing zero-probability issues with Laplace Smoothing.
  * **Implementation:** Custom `NaiveBayesSpamFilter` class tested on spam/ham email datasets.
* **Week 8: The Art of Evaluation**
  * Beyond raw accuracy: handling class imbalance.
  * Implementing a custom Confusion Matrix calculator from scratch (`TP, FP, TN, FN`), Precision, Recall, F1-Score, and analyzing clinical/spam decision thresholds.
* **Week 9: Integration & Synthesis**
  * Summary of the Bayesian Advantage: Small-data robustness, interpretability, and natural online learning capabilities.

---

## 💻 Tech Stack & Requirements
* **Language:** Python 3.x
* **Libraries:** `numpy`, `scipy`, `matplotlib`, `pandas`

To install dependencies:
```bash
pip install numpy scipy matplotlib pandas