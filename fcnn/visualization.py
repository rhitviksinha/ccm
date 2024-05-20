import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def plot_rep(rep1, rep2, rep3, names, nepochs_list):
    nrows = len(names)
    R = np.dstack((rep1, rep2, rep3))
    mx = R.max()
    mn = R.min()
    depth = R.shape[2]
    count = 1
    plt.figure(1, figsize=(4.2, 8.4))
    for i in range(nrows):
        for d in range(depth):
            plt.subplot(nrows, depth, count)
            rep = R[i, :, d]
            plt.bar(range(rep.size), rep)
            plt.ylim([mn, mx])
            plt.xticks([])
            plt.yticks([])
            if d == 0:
                plt.ylabel(names[i])
            if i == 0:
                plt.title(f"epoch {nepochs_list[d]}")
            count += 1
    plt.show()

def plot_dendo(rep1, rep2, rep3, names, nepochs_list):
    plt.figure(2, figsize=(7, 12))
    for i, rep in enumerate([rep1, rep2, rep3]):
        linked = linkage(rep, 'single')
        plt.subplot(3, 1, i + 1)
        dendrogram(linked, labels=names, color_threshold=0)
        plt.ylim([0, np.max(linked[:, 2]) + 0.1])
        plt.title(f"Hierarchical clustering; epoch {nepochs_list[i]}")
        plt.ylabel('Euclidean distance')
    plt.show()
