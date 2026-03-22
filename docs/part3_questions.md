# Part 3 Interview Questions

## Question 1 (ANN)
**Question:** Explain how backpropagation updates weights in a neural network.

**Model Answer:**
Backpropagation computes gradients of the loss function with respect to each weight using the chain rule. These gradients are used in gradient descent to update weights and minimize error.

---

## Question 2 (CNN)
**Question:** Why is pooling used in CNNs and what problem does it solve?

**Model Answer:**
Pooling reduces spatial dimensions, decreases computation, and provides translation invariance. It helps prevent overfitting and captures dominant features.

---

## Question 3 (Cross-topic: RNN + LSTM)
**Question:** How do LSTMs improve upon standard RNNs?

**Model Answer:**
LSTMs solve the vanishing gradient problem in RNNs by using a cell state and gating mechanisms (input, forget, output gates) that control information flow over long sequences.