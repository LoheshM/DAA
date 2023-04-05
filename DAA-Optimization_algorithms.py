#!/usr/bin/env python
# coding: utf-8

# In[199]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report,log_loss
import copy


# In[200]:


data = pd.read_csv(r"C:\Users\Lohesh\Downloads\Notes\Machine Learning\codes\Bank_Personal_Loan_Modelling.csv")


# In[201]:


data.head()


# In[202]:


data.info()
data.drop(['ID','ZIP Code'],axis=1,inplace=True)


# In[203]:


data = data[data['Income']<=160]


# In[204]:


X = data.drop('Personal Loan',axis=1).values
Y = data['Personal Loan'].values


# In[205]:


xtr,xtst,ytr,ytst = tts(X,Y,test_size=0.25,stratify=Y,random_state=42)
xtr.shape,xtst.shape,ytr.shape,ytst.shape


# In[206]:


sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xtst = sc.transform(xtst)


# In[250]:


class Basic_nn:
    def __init__(self,ip_size):
        self.weightstruct = []
        self.ip_size = ip_size 
        self.act = []
    
    def loss_fn(pred,op):
        return sum(abs(pred.flatten()-op.flatten()))
    
    def weights(self):
        new = []
        for i in self.weightstruct:
            new.append(np.random.random(i)*3-0.15)
        return new
        
    def layer_add(self,layer_size,activation):
        self.act.append(activation)
        if self.weightstruct == []:
            self.weightstruct.append((self.ip_size+1,layer_size))
        else:
            prev = self.weightstruct[-1][-1]
            self.weightstruct.append((prev+1,layer_size))
            
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    def ReLU(x):
        return (x>0)*x
    
    def for_pro(self,ip,weights):
        layer = ip
        for i,j in zip(weights,self.act):
            layer = np.append(np.ones((len(layer),1)),layer,axis=1)
            if j=='sigmoid':
                layer = Basic_nn.sigmoid(layer @ i)
            else:
                layer = Basic_nn.ReLU(layer @ i)
        return layer
    


# In[251]:


initial = Basic_nn(11) 
initial.layer_add(16,'ReLU') 
initial.layer_add(8,'sigmoid')
initial.layer_add(1,'sigmoid')
initial.weightstruct
initial.for_pro(xtr,initial.weights())


# # Genetic Algorithm adjusting Neural Network Weights

# In[252]:


class gene_alg:
    def __init__(self,initial,pop_size):
        self.initial = initial
        self.pop_size = pop_size
        self.population = [initial.weights() for i in range(pop_size)]
        self.gen = 0
        
    def off_spr(p1,p2):
        c1 = []
        c2 = []
        for i in range(len(p1)):
            a = p1[i]
            b = p2[i]
            
            ind = np.random.randint(0,2,a.shape)
            cd1 = a*ind + b*(1^ind)
            cd2 = a*(1^ind) + b*ind
            
            cd1 += (np.random.choice([0,1],cd1.shape,p=[0.7,0.3]) * np.random.random(cd1.shape)*2-0.3)
            cd1 += (np.random.choice([0,1],cd1.shape,p=[0.7,0.3]) * np.random.random(cd2.shape)*2-0.3)
            
            c1.append(cd1)
            c2.append(cd2)
        return c1,c2
        
    def create(self,xtr,ytr):
        nextgen = self.population.copy()
        for i in range(self.pop_size-1):
            for j in range(i,self.pop_size):
                p1 = self.population[i]
                p2 = self.population[j]
                c1,c2 = gene_alg.off_spr(p1,p2)
                nextgen.append(c1)
                nextgen.append(c2)
                
        sortedgen = sorted(nextgen,key = lambda w : Basic_nn.loss_fn(initial.for_pro(xtr,w),ytr))
        self.population = sortedgen[:self.pop_size]
        
        self.gen += 1
        print("loss_value: %f & gene_number: %d"%(Basic_nn.loss_fn(initial.for_pro(xtr,self.population[0]),ytr),self.gen))
                


# In[253]:


gen = gene_alg(initial,10)


# In[254]:


for i in range(50):
    gen.create(xtr,ytr)


# In[255]:


print(classification_report(ans,ytst))


# In[256]:


ans = (initial.for_pro(xtst,gen.population[0])>0.5).astype(int)
confusion_matrix(ans,ytst)


# # Adjustment of weights using Cultural Algorithm

# In[266]:


class cul_alg(gene_alg):
    def __init__(self,initial,pop_size):
        self.initial = initial
        self.pop_size = pop_size
        self.population = [base.weights() for i in range(pop_size)]
        self.gen = 0
        self.pb = copy.deepcopy(self.population)
        self.gb = self.population[0]
        
        self.a = 0.9
        self.b = 0.05
        self.c = 0.05
        self.decay_rate = 0.01
        
    def create(self,xtr,ytr):
        nextgen = self.population.copy()
        for i in range(self.pop_size-1):
            for j in range(i,self.pop_size):
                p1 = self.population[i]
                p2 = self.population[j]
                c1,c2 = gene_alg.off_spr(p1,p2)
                nextgen.append(c1)
                nextgen.append(c2)
                
        sortedgen = sorted(nextgen,key = lambda w : Basic_nn.loss_fn(initial.for_pro(xtr,w),ytr))
        self.population = sortedgen[:self.pop_size//2]
        
        cfit = max(self.population,key = lambda w : self.c_score(w,xtr,ytr))
        self.population += self.c_influence(self.population,cfit)
        
        self.gen += 1
        print(" loss: %f & generation: %d "%(nn_base.loss(base.forward(xtr,self.population[0]),ytr),self.gen))
        
    def c_score(self,w,xtr,ytr):
        pred = base.for_pro(xtr,w)
        a = sum(ytr==1) #True count
        b = sum(pred[np.where(ytr==1)]>0.5) #True positive count
        return b/a
    
    def c_influence(self,pop,best):
        pop1 = copy.deepcopy(pop)
        for w in pop1:
            for j in range(len(w)):
                w[j] += np.random.choice([0,1],w[j].shape,p=[0.8,0.2]) * best[j]
        return pop1


# In[267]:


new = cul_alg(base,10)


# In[268]:


for i in range(40):
    new.create(xtr,ytr)


# In[ ]:


ans = (initial.for_pro(xtst,new.population[0])>0.5).astype(int)
confusion_matrix(ans,ytst)


# In[ ]:


print(classification_report(ans,ytst))


# # Particle swarm Optimiztaion on Neural Network

# In[269]:


class swarm:
    def __init__(self,initial,pop_size):
        self.initial = initial
        self.pop_size = pop_size
        self.population = [base.weights() for i in range(pop_size)]
        self.gen = 0
        self.pb = copy.deepcopy(self.population)
        self.gb = self.population[0]
        
        self.a = 0.9
        self.b = 0.05
        self.c = 0.05
        self.decay_rate = 0.01
        
    def param_upd(self):
        self.a-=(self.a*self.decay_rate)
        
    def best_upd(self,xtr,ytr):
        loss = lambda w : Basic_nn.loss_fn(initial.for_pro(xtr,w),ytr)
        
        for ind in range(self.pop_size):
            w = self.population[ind]
            
            if loss(w) < loss(self.pb[ind]):
                self.pb[ind] = copy.deepcopy(w)
                
            if loss(self.pb[ind]) < loss(self.gb):
                self.gb = copy.deepcopy(self.pb[ind])
                
        print(" loss_value: %f & gene_number: %d"%(loss(self.gb),self.gen))
        
    def upd(self):
        for ind in range(self.pop_size):
            w = self.population[ind]
            pb = self.pb[ind]
            for i in range(len(w)):
                r1, r2 = np.random.rand(2)
                w[i] = self.a*w[i] + self.b*r1*(pb[i]-w[i]) + self.c*r2*(self.gb[i]-w[i])
                
    def create(self,xtr,ytr):
        self.upd()
        self.best_upd(xtr,ytr)
        self.gen += 1
        if self.gen%10==0:
            self.param_upd()


# In[276]:


swarm_base = Basic_nn(11)
swarm_base.layer_add(16,'ReLU')
swarm_base.layer_add(8,'sigmoid')
swarm_base.layer_add(1,'sigmoid')
swarm_base.weightstruct


# In[277]:


particle = swarm(initial,100)


# In[278]:


for i in range(20):
    particle.create(xtr,ytr)


# In[279]:


ans = (initial.for_pro(xtst,particle.gb)>0.5).astype(int)
confusion_matrix(ans,ytst)


# In[280]:


print(classification_report(ans,ytst))


# # ANT colony Optimization for weight adjustment

# In[286]:


import numpy as np

class Ant:
    def __init__(self, num_weights):
        self.weights = np.random.rand(num_weights)
        self.fitness = 0.0

    def evaluate_fitness(self, fitness_func):
        self.fitness = fitness_func(self.weights)

class ACO:
    def __init__(self, pop_size, num_weights, q, alpha, beta, rho, num_iterations):
        self.pop_size = pop_size
        self.num_weights = num_weights
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.num_iterations = num_iterations
        self.pheromone = np.ones(num_weights)
        self.best_ant = None
        self.best_fitness = float('inf')
        self.population = [Ant(num_weights) for _ in range(pop_size)]

    def update_pheromone(self, ant):
        self.pheromone *= (1.0 - self.rho)
        self.pheromone += (self.q / ant.fitness) * ant.weights

    def select_weights(self):
        weights = np.zeros(self.num_weights)
        for i in range(self.num_weights):
            if np.random.rand() < self.pheromone[i]:
                weights[i] = 1
        return weights

    def update_population(self, fitness_func):
        for ant in self.population:
            weights = self.select_weights()
            ant.weights = weights
            ant.evaluate_fitness(fitness_func)
            if ant.fitness < self.best_fitness:
                self.best_fitness = ant.fitness
                self.best_ant = ant
            self.update_pheromone(ant)

    def evolve(self, fitness_func):
        for i in range(self.num_iterations):
            self.update_population(fitness_func)
        return self.best_ant.weights
    
    def fitness_func(self,weights):
        model.set_weights(weights)
        model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return -loss


# In[287]:


pop_size = 100
num_weights = 337
q = 0.1
alpha = 1.0
beta = 2.0
rho = 0.5
num_iterations = 100

aco = ACO(pop_size, num_weights, q, alpha, beta, rho, num_iterations)

optimal_weights = aco.evolve(fitness_func)


# In[ ]:




