import matplotlib
# Force matplotlib to not use Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle

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

    plt.legend(loc='upper left')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)

# Loads the pickle file, and then calls plot_many
def plot_from_pickle(file_name, figname):
    f = open(file_name, "rb")
    result_dict = pickle.load(f)
    f.close()
    xs = result_dict['xs']
    ys = result_dict['ys']
    labels = result_dict['labels']
    xlabel = result_dict['xlabel']
    ylabel = result_dict['ylabel']
    title = result_dict['title']
    plot_many(xs, ys, labels=labels,
    figname=figname, xlabel=xlabel, ylabel=ylabel,
    title=title)
