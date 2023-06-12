from AgentBasedModel.agents import *
from AgentBasedModel.events import *
from AgentBasedModel.simulator import *
from AgentBasedModel.visualization import *
from AgentBasedModel.states import *
from AgentBasedModel import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN, AffinityPropagation, MeanShift, OPTICS, KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score


# Simulation
exchange = ExchangeAgent(volume=1000)
simulator = Simulator(**{
    'exchange': exchange,
    'traders': [Random(exchange, 10**3) and Chartist(exchange, 10**3) and Fundamentalist(exchange, 10**3) and
                Universalist(exchange, 10**3) for _ in range(20)],
    'events': [MarketPriceShock(100, -10)]
})
info = simulator.info
simulator.simulate(1005)


# States analyse
list_states = status(info, 5)
print(list_states)
df1 = pd.DataFrame(list_states, columns=['states'])
df2 = pd.get_dummies(df1, columns=['states'])
# print('df2 =', len(df2))

col1 = df2['states_calm-down'].tolist()
col2 = df2['states_calm-up'].tolist()
col3 = df2['states_panic'].tolist()
col4 = df2['states_recovery'].tolist()

line = []
cnt = -1

l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
l7 = []
l8 = []
l9 = []
l10 = []
l11 = []
l12 = []
l13 = []
l14 = []
l15 = []
l16 = []

for i in range(len(col1)):
    cnt += 1
    el1 = col1[i]
    el2 = col2[i]
    el3 = col3[i]
    el4 = col4[i]
    if cnt != 4:
        line.append(el1)
        line.append(el2)
        line.append(el3)
        line.append(el4)
    else:
        # print(line)
        cnt = -1
        l1.append(line[0])
        l2.append(line[1])
        l3.append(line[2])
        l4.append(line[3])
        l5.append(line[4])
        l6.append(line[5])
        l7.append(line[6])
        l8.append(line[7])
        l9.append(line[8])
        l10.append(line[9])
        l11.append(line[10])
        l12.append(line[11])
        l13.append(line[12])
        l14.append(line[13])
        l15.append(line[14])
        l16.append(line[15])
        line = []

new_df = pd.DataFrame()
new_df['states_calm-down1'] = l1
new_df['states_calm-up1'] = l2
new_df['states_panic1'] = l3
new_df['states_recovery1'] = l4
new_df['states_calm-down2'] = l5
new_df['states_calm-up2'] = l6
new_df['states_panic2'] = l7
new_df['states_recovery2'] = l8
new_df['states_calm-down3'] = l9
new_df['states_calm-up3'] = l10
new_df['states_panic3'] = l11
new_df['states_recovery3'] = l12
new_df['states_calm-down4'] = l13
new_df['states_calm-up4'] = l14
new_df['states_panic4'] = l15
new_df['states_recovery4'] = l16
# print('new_df =', len(new_df))


intr = interpreter(info)
pct_final = intr[0]
iter_final = intr[1]

itera = list()
for i in range(1, 1005):
    itera.append(i)

# sns.scatterplot(x = pct_final, y = iter_final)

df = pd.DataFrame()
df['pct_change'] = pct_final
df['iter'] = iter_final

indlist = list(df.index.values)

# Pattern searching
'''''
# MeanShift
msh = MeanShift()
msh.fit(new_df)
# Number of Clusters
labels = msh.labels_
N_clus = len(set(labels))
print('Estimated no. of clusters: %d' % N_clus)
print(msh.labels_)

# Plot scatter with clusters
plt.scatter(msh.labels_,
            itera[:39],
            c=msh.labels_)
plt.show()
'''''

'''''
# OPTICS
op = OPTICS()
# min_samples=6, xi=0.015
op.fit(new_df)
# Number of Clusters
labels = op.labels_
N_clus = len(set(labels))
print('Estimated no. of clusters: %d' % N_clus)
print(op.labels_)
print(len(op.labels_))

# Plot scatter with clusters
plt.scatter(op.labels_,
            itera[:40],
            c=op.labels_)
plt.show()
'''''


# AffinityPropagation
ap = AffinityPropagation(damping=0.65, max_iter=300, convergence_iter=15)
# for price df (damping=0.65, max_iter=300, convergence_iter=15)
# for states df2 (damping=0.79, max_iter=300, convergence_iter=30)
ap.fit(df)

# Number of Clusters
labels = ap.labels_
N_clus = len(set(labels))
print('Estimated no. of clusters: %d' % N_clus)
print(ap.labels_)

'''''
# For states
for i in range(0, len(ap.labels_)):
    print(5+20*i, '-', 25+20*i, '= cluster', ap.labels_[i], 'states:', list_states[i*4], list_states[i*4+1],
          list_states[i*4+2], list_states[i*4+3])
'''''

# Plot scatter with clusters
plt.scatter(ap.labels_,
            itera[:len(ap.labels_)],
            c=ap.labels_)
plt.show()


'''''
# Kmeans
km = AgglomerativeClustering(n_clusters=24)      # 4! as we grope 4 states in 1 big_state
km.fit(new_df)  # df for prices; new_df for states
# Number of Clusters
labels = km.labels_
N_clus = len(set(labels))
print('Estimated no. of clusters: %d' % N_clus)
print(km.labels_)

# For states
for i in range(0, len(km.labels_)):
    print(5+20*i, '-', 25+20*i, '= cluster', km.labels_[i], 'states:', list_states[i*4], list_states[i*4+1],
          list_states[i*4+2], list_states[i*4+3])

# Plot scatter with clusters
plt.scatter(km.labels_,
            itera[:len(km.labels_)],
            c=km.labels_)
plt.show()
'''''

'''''
# DBSCAN
dbscan_cluster = DBSCAN(eps=2.0001, min_samples=4)
dbscan_cluster.fit(new_df)

# Number of Clusters DBSCAN
labels = dbscan_cluster.labels_
N_clus=len(set(labels))-(1 if -1 in labels else 0)
print('Estimated no. of clusters: %d' % N_clus)
# Identify Noise
n_noise = list(dbscan_cluster.labels_).count(-1)
print('Estimated no. of noise points: %d' % n_noise)
# Calculating v_measure
print('v_measure =', v_measure_score(indlist, labels))

print(dbscan_cluster.labels_)

# Plot scatter with clusters
plt.scatter(dbscan_cluster.labels_,
            itera,
            c=dbscan_cluster.labels_)
plt.show()
'''''


# For Prices
clust_list = list(ap.labels_)
seq_list = []
iter_for_seq = []
seq = []
it_seq = []
el0 = clust_list[0]
for i in range(1, len(clust_list)):
    if len(seq) == 0:
        it_seq.append(i-1)
        seq.append(el0)
    el0 = clust_list[i-1]
    el1 = clust_list[i]
    if el0 == el1:
        seq.append(el1)
        it_seq.append(i)
    else:
        seq_list.append(seq)
        iter_for_seq.append(it_seq)
        seq = []
        it_seq = []
        el0 = el1

n_seq = 2
dict_seq = {}
dict_seq_iter = {}
for i in range(25):
    n_seq += 1
    seq_list_fixed_len = []
    iter_for_seq_list_fixed_len = []
    for i in range(len(seq_list)):
        seq = seq_list[i]
        iter_seq = iter_for_seq[i]
        if len(seq) == n_seq:
            seq_list_fixed_len.append(seq[0])
            iter_for_seq_list_fixed_len.append([iter_seq[0], iter_seq[len(iter_seq)-1]])
        if len(seq_list_fixed_len) > 1 and len(seq_list_fixed_len) != len(set(seq_list_fixed_len)):
            dict_seq[n_seq] = seq_list_fixed_len #sorted
            dict_seq_iter[n_seq] = iter_for_seq_list_fixed_len

seq_keys = list(dict_seq.keys())
k1 = seq_keys[len(seq_keys)-1]
k2 = seq_keys[len(seq_keys)-2]

from collections import Counter

def positions_to_remain(list1):
    count = dict(Counter(list1))
    selected_val = []
    for i in count:
        if count[i] > 1:
            selected_val.append(i)
    remain_list = []
    for i in range(len(list1)):
        if list1[i] in selected_val:
            remain_list.append(i)
    return remain_list

def extract_values(list1, pos):
    extracted_list = []
    for i in range(len(list1)):
        if i in pos:
            extracted_list.append(list1[i])
    return extracted_list

pos_k1 = positions_to_remain(dict_seq[k1])
pos_k2 = positions_to_remain(dict_seq[k2])

# print(str(k1)+':', dict_seq[k1], str(k2)+':', dict_seq[k2])
# print(str(k1)+':', dict_seq_iter[k1], str(k2)+':', dict_seq_iter[k2])

print(str(k1)+':', extract_values(dict_seq[k1], pos_k1), str(k2)+':', extract_values(dict_seq[k2], pos_k2))
print(str(k1)+':', extract_values(dict_seq_iter[k1], pos_k1), str(k2)+':', extract_values(dict_seq_iter[k2], pos_k2))


plot_volatility_price(info)
plot_price_fundamental(info)

