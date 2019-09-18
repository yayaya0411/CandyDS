import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

"""
set matplotlib figure show traditional chinese
move front.ttf into matplotlib fonts path
"C:\\ProgramData\\Anaconda3\\Lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf"
"""
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

def values(ax):
    """
    annotate values on figure

    Args:
    ----------
        ax : figure ax
    """
    for i in ax.patches:
        x = i.get_bbox().get_points()[:, 0]
        y = i.get_height()
        ymin, ymax = ax.get_ylim()
        move = (ymax - ymin) * 0.02
        ax.annotate(y, (x.mean(), y + move), ha='center', va='center', size=12)


def plot_obj(data, cols, drops=[], output=False):
    """
    quick review object data by countplot and save picture single/mutiple

    Args:
    ----------
        data : dataframe
        drops : drop object columns that you don't want plot
        cols : contral figure cols
        output : output single column picture
    """
    output_folder = os.path.join('.', 'Picture')
    data_tmp = data.select_dtypes(include=["object"])
    data_tmp = data_tmp.drop(columns=drops)
    loop_cols = data_tmp.columns
    pic_format = '.png'
    if os.path.isdir(output_folder) == False:
        os.mkdir(output_folder)
    if output:
        for i in range(len(loop_cols)):
            fig, ax = plt.subplots(1, 1, figsize=(16, 12))
            sns.countplot(x=loop_cols[i], data=data_tmp, ax=ax)
            fig.savefig(os.path.join(output_folder,
                                     'obj_' + loop_cols[i] + pic_format))
    fig_rows = math.ceil(data_tmp.shape[1] / cols)
    fig, ax = plt.subplots(fig_rows, cols, figsize=(cols * 8, fig_rows * 6))
    counter = 0
    print(str(cols) + ' x ' + str(fig_rows) + ' figure ' +
          str(data_tmp.shape[1]) + ' Variable ')
    print(loop_cols)
    if fig_rows == 1 or cols == 1:
        for i in range(max(fig_rows, cols)):
            try:
                orders = data_tmp.iloc[:, counter].value_counts().index
                sns.countplot(x=loop_cols[counter],
                              data=data_tmp, ax=ax[i], order=orders)
                plt.xticks(rotation=45)
                values(ax[i])
                print(str(i) + ' plot ' + loop_cols[counter])
                counter += 1
            except:
                pass
        fig.savefig(os.path.join(output_folder, 'plot_obj' + pic_format))
    else:
        for i in range(0, fig_rows):
            for j in range(0, cols):
                try:
                    orders = data_tmp.iloc[:, counter].value_counts().index
                    sns.countplot(
                        x=loop_cols[counter], data=data_tmp, ax=ax[i, j], order=orders)
                    plt.xticks(rotation=45)
                    values(ax[i, j])
                    print(str(i) + ',' + str(j) +
                          ' plot ' + loop_cols[counter])
                    counter += 1
                except IndexError:
                    print(str(i) + ',' + str(j) + ' no more col to plot')
                    pass
        fig.savefig(os.path.join(output_folder, 'plot_obj' + pic_format))


def plot_int(data, cols, drops=[], bins=10, kde=False, output=False):  # not yet
    """
    quick review int/float data by distplot and save picture single/mutiple

    Args:
    ----------
        data : dataframe
        cols : contral figure columns
        drops : drop int/float columns that you don't want plot
        bins : automatic grouping of bins
        kde : kernel density estimation
        output : output single column picture
    """
    output_folder = os.path.join('.', 'Picture')
    data_tmp = data.select_dtypes(include=["float64", "int"])
    data_tmp = data_tmp.drop(columns=drops)
    loop_cols = data_tmp.columns
    pic_format = '.png'
    if os.path.isdir(output_folder) == False:
        os.mkdir(output_folder)
    if output:
        for i in range(len(loop_cols)):
            fig, ax = plt.subplots(1, 1, figsize=(16, 12))
            a = loop_cols[i]
            sns.distplot(data_tmp[a].dropna(), bins=bins, kde=kde, ax=ax)
            fig.savefig(os.path.join(output_folder,
                                     'int_' + loop_cols[i] + pic_format))
    fig_rows = math.ceil(data_tmp.shape[1] / cols)
    fig, ax = plt.subplots(fig_rows, cols, figsize=(cols * 8, fig_rows * 6))
    counter = 0
    print(str(cols) + ' x ' + str(fig_rows) + ' figure ' +
          str(data_tmp.shape[1]) + ' Variable ')
    print(loop_cols)
    if fig_rows == 1 or cols == 1:
        for i in range(max(fig_rows, cols)):
            try:
                a = loop_cols[counter]
                sns.distplot(data_tmp[a].dropna(),
                             bins=bins, kde=kde, ax=ax[i])
                plt.xticks(rotation=45)
                print(str(i) + ' plot ' + loop_cols[counter])
                counter += 1
            except:
                pass
        fig.savefig(os.path.join(output_folder, 'plot_int' + pic_format))
    else:
        for i in range(0, fig_rows):
            for j in range(0, cols):
                try:
                    a = loop_cols[counter]
                    sns.distplot(data_tmp[a].dropna(),
                                 bins=bins, kde=kde, ax=ax[i, j])
                    plt.xticks(rotation=45)
                    print(str(i) + ',' + str(j) +
                          ' plot ' + loop_cols[counter])
                    counter += 1
                except IndexError:
                    print(str(i) + ',' + str(j) + ' no more col to plot')
                    pass
        fig.savefig(os.path.join(output_folder, 'plot_int' + pic_format))
