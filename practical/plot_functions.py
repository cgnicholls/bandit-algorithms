import matplotlib.pyplot as plt

def plot(xs, ys, xlabel="", ylabel="", title="", figname=""):
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)

def plot_many(xs, ys_list, labels=None, xlabel="", ylabel="", title="", figname=""):
    fig, ax = plt.subplots()
    if labels is None:
        labels=["" for i in range(len(ys_list))]
    handles = []
    for i, ys in enumerate(ys_list):
        plot = ax.plot(xs, ys, label=labels[i])
        handles.append(plot)

    plt.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
