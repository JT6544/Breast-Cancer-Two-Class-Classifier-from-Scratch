import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from matplotlib import rcParams 
rcParams['figure.autolayout'] = True 

data = np.loadtxt('breast_cancer_data.csv', delimiter=',') 
x = data[:-1, :] 
y = data[-1:, :] 
unique = np.unique(y) # Mapping from 2=benign, 4=malignant to -1=benign, +1=malignant 
if set(unique) == {2., 4.}:
    y = np.where(y == 4, 1.0, -1.0)
else:
    y = np.where(y == np.max(unique), 1.0, -1.0)
    
P = x.shape[1] 

print("P =", P)
print("Mapped label values:", np.unique(y))
print("Class counts (Malignant +1, Benign -1):", int(np.sum(y.flatten() == 1.0)), int(np.sum(y.flatten() == -1.0)))

def model(x, w):
    a = w[0] + np.dot(x.T, w[1:])
    return a.T

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def softmax(w):
    cost = np.sum(np.log(1 + np.exp(-y * model(x, w))))
    return cost / float(np.size(y))

def counting_cost(w, x_local, y_local):
    y_hat = np.sign(model(x_local, w)) 
    y_hat = np.where(y_hat == 0, 1, y_hat) 
    ind = np.argwhere(y_local != y_hat) 
    ind = [v[1] for v in ind]
    total = len(ind)
    mal = int(np.sum((y_local.flatten() == 1.0) & (y_hat.flatten() != 1.0))) 
    ben = int(np.sum((y_local.flatten() == -1.0) & (y_hat.flatten() != -1.0))) 
    return total, mal, ben, y_hat

def gradient_descent(g, alpha_choice, max_its, w):
    gradient = grad(g)
    weight_history = [w]
    cost_history = [g(w)]
    for k in range(max_its):
        if alpha_choice == 'diminishing':
            alpha = 1 / float(k + 1)
        else:
            alpha = alpha_choice
        grad_eval = gradient(w)
        w = w - alpha * grad_eval
        weight_history.append(w)
        cost_history.append(g(w))
    return weight_history, cost_history

N = 8
np.random.seed(0)
w = 0.1 * np.random.randn(N + 1, 1)
max_its = 1000
alpha = 0.1
g = softmax

print("max_its =", max_its, ", Alpha =", alpha, ", Cost Function =", g.__name__) 
weight_history, cost_history = gradient_descent(g, alpha, max_its, w)

Klen = len(weight_history) 
total_mis = np.zeros(Klen, dtype=int) 
mis_mal = np.zeros(Klen, dtype=int) 
mis_ben = np.zeros(Klen, dtype=int)  

for i, w_iter in enumerate(weight_history): 
    mt, mm, mb, _ = counting_cost(w_iter, x, y)
    total_mis[i] = mt
    mis_mal[i] = mm
    mis_ben[i] = mb

k_best = int(np.argmin(total_mis)) 

print("\nBest iteration k_best =", k_best)
print("Total misclassifications at k_best =", int(total_mis[k_best]))
print("Misclassifications at k_best: Malignant =", int(mis_mal[k_best]), ", Benign =", int(mis_ben[k_best]))
print("Total misclassifications at k_max =", int(total_mis[-1]))
print("Misclassifications at k_max: Malignant =", int(mis_mal[-1]), ", Benign =",int(mis_ben[-1]))

w_best = weight_history[k_best]
_, _, _, y_hat_best = counting_cost(w_best, x, y)
y_true = y.flatten()

TP = int(np.sum((y_true == 1.0) & (y_hat_best.flatten() == 1.0)))   
FN = int(np.sum((y_true == 1.0) & (y_hat_best.flatten() != 1.0)))  
FP = int(np.sum((y_true != 1.0) & (y_hat_best.flatten() == 1.0)))   
TN = int(np.sum((y_true != 1.0) & (y_hat_best.flatten() != 1.0)))   

P_plus = int(np.sum(y_true == 1.0))
P_minus = int(np.sum(y_true == -1.0))

print("\nConfusion matrix at k_best (Positive = Malignant):")
print("            Pred +     Pred -")
print(f"Actual +   TP = {TP}   FN = {FN}")
print(f"Actual -   FP = {FP}   TN = {TN}")

accuracy = (TP + TN) / float(P)
acc_plus = TP / float(P_plus)
acc_minus = TN / float(P_minus)
balanced_acc = 0.5 * (acc_plus + acc_minus)

print(f"\nPopulation sizes: P = {P}, P_plus (Malignant) = {P_plus}, P_minus (Benign) = {P_minus}")
print(f"Overall Accuracy (TP+TN)/P = {accuracy:.4f}")
print(f"Accuracy Malignant (TP/P_plus) = {acc_plus:.4f}")
print(f"Accuracy Benign (TN/P_minus) = {acc_minus:.4f}")
print(f"Balanced Accuracy = {balanced_acc:.4f}")

print("\nOptimal weights w_best (bias first):") 
for i, val in enumerate(np.array(weight_history[k_best]).flatten()):
    print(f"w[{i}] = {val:.6f}")

xA = np.array([3, 5, 2, 7, 6, 1, 5, 1]).reshape(8, 1)
xB = np.array([1, 5, 4, 7, 1, 6, 5, 2]).reshape(8, 1)
xC = np.array([1, 9, 1, 1, 1, 9, 1, 1]).reshape(8, 1)

def predict_and_prob(w, x_vec):
    val = float((w[0] + np.dot(x_vec.T, w[1:]))[0, 0]) 
    prob_malignant = 1.0 / (1.0 + np.exp(-val)) 
    pred = 1 if val >= 0 else -1 
    return val, pred, prob_malignant

for name, xv in [('A', xA), ('B', xB), ('C', xC)]: 
    val, pred, prob = predict_and_prob(w_best, xv)
    print(f"\nPatient {name}: Linear Score = {val:.6f}, Predicted = {'Malignant (+1)' if pred==1 else 'Benign (-1)'}")
    print(f"P(Malignant) = {prob:.6f}, P(Benign) = {1.0-prob:.6f}")

k_axis = np.arange(Klen) 

plt.figure(figsize=(8, 4))
plt.plot(k_axis, cost_history)
plt.xlabel('Iteration k')
plt.ylabel('Cost')
plt.title('Cost vs k')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(k_axis, total_mis)
plt.xlabel('Iteration k')
plt.ylabel('Total Misclassifications')
plt.title('Total Misclassifications vs k')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(k_axis, mis_mal, label='Malignant')
plt.plot(k_axis, mis_ben, label='Benign')
plt.xlabel('k')
plt.ylabel('Misclassifications')
plt.title('Misclassifications by class vs k')
plt.legend()
plt.grid(True)
plt.show()

num_steps = len(weight_history)
num_weights = len(weight_history[0])  
w_array = np.zeros((num_steps, num_weights))

for i, weights in enumerate(weight_history): 
    w_array[i, :] = np.array(weights).flatten()
    
plt.figure(figsize=(8, 5))

for j in range(num_weights):
    plt.plot(k_axis, w_array[:, j], label=f"w[{j}]")
    
plt.xlabel('k')
plt.ylabel('Weight Value')
plt.title('Weight Evolution (w[0]=Bias)')
plt.legend(ncol=3, fontsize='small')
plt.grid(True)
plt.show()

k_max_early = min(101, Klen)  
k_axis_early = np.arange(k_max_early)

plt.figure(figsize=(8, 4))
plt.plot(k_axis_early, cost_history[:k_max_early]) 
plt.xlabel('Iteration k')
plt.ylabel('Cost')
plt.title('Cost vs k (First 100 Iterations)')
plt.grid(True)
plt.xlim(0, k_max_early-1)  
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(k_axis_early, total_mis[:k_max_early])  
plt.xlabel('Iteration k')
plt.ylabel('Total Misclassifications')
plt.title('Total Misclassifications vs k (First 100 Iterations)')
plt.grid(True)
plt.xlim(0, k_max_early-1)  
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(k_axis_early, mis_mal[:k_max_early], label='Malignant')  
plt.plot(k_axis_early, mis_ben[:k_max_early], label='Benign')
plt.xlabel('k')
plt.ylabel('Misclassifications')
plt.title('Misclassifications by class vs k (First 100 Iterations)')
plt.legend()
plt.grid(True)
plt.xlim(0, k_max_early-1)
plt.show()

num_steps = len(weight_history)
num_weights = len(weight_history[0])  
w_array = np.zeros((num_steps, num_weights))

for i, weights in enumerate(weight_history):
    w_array[i, :] = np.array(weights).flatten()
    
plt.figure(figsize=(8, 5))
for j in range(num_weights):
    plt.plot(k_axis_early, w_array[:k_max_early, j], label=f"w[{j}]")
    
plt.xlabel('k')
plt.ylabel('Weight Value')
plt.title('Weight Evolution (w[0]=Bias) - First 100 Iterations')
plt.legend(ncol=3, fontsize='small')
plt.grid(True)
plt.xlim(0, k_max_early-1)
plt.show()
