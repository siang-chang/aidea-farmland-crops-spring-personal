#!/usr/bin/env python
# coding: utf-8
import math
from typing import List, Union

import cv2
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from beartype import beartype
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

from .image import resize_with_aspect_ratio


class ShowPlotMode:
    SHOW_WITH_JS = None
    SHOW_STATIC_PLOT = 'png'


show_mode = ShowPlotMode.SHOW_WITH_JS


def plot_pred_img(
    img_path: str, true_label: Union[str, int], pred_label: Union[str, int],
) -> None:
    """Plot a image with true label and predict label.

    Args:
        img_path (str): Image file path.
        true_label (Union[str, int]): Image ground truth label.
        pred_label (Union[str, int]): Image predict label.
    """
    title = f"{img_path}<br>true: {true_label} / predict: {pred_label}"
    plot_img(img_path, title)


def plot_img(img_path: str, title: str, return_fig: bool = False) -> Union[None, plotly.graph_objs._figure.Figure]:
    """Plot a image.

    Args:
        img_path (str): Image file path.
        title (str): Title of the plot.
        return_fig (bool, optional): If True, return Plotly Figure object. Defaults to False.

    Returns:
        Union[None, plotly.graph_objs._figure.Figure]: Plotly figure object with image.
    """
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # resize the image to a smaller one
    # img = resize_with_aspect_ratio(img, width=1280)

    fig = px.imshow(img, labels=dict(x=title))
    if return_fig:
        return fig
    fig.update_xaxes(showticklabels=False, side="top")
    fig.update_yaxes(showticklabels=False)
    fig.show(show_mode)


@beartype
def plot_confusion_matrix(
        y_true: List[Union[int, str]], y_pred: List[Union[int, str]],
        labels: List[Union[int, str]], save_path: str = None
) -> None:
    """Calculate and plot confusion matrix, and then save the plot to image file.

    Args:
        y_true (List[Union[int, str]]): The list of ground truth labels.
        y_pred (List[Union[int, str]]): The list of predict labels.
        labels (List[Union[int, str]]): All unique labels.
        save_path (str, optional): The file path you want to save the confusion matrix plot to. Defaults to None.
    """
    conf_matrix = confusion_matrix(
        y_true, y_pred, normalize="true", labels=labels
    )
    title = '<i><b>Confusion matrix</b></i>'
    x_title = "Predicted label"
    y_title = "True label"
    plot_matrix(conf_matrix, title, labels, labels,
                x_title, y_title, save_path=save_path)


def plot_matrix(
    matrix_values: np.array, title: str,
    x_labels: List[Union[int, str]], y_labels: List[Union[int, str]],
    x_title: str, y_title: str, colorscale: str = 'Mint', save_path: str = None, num_of_digits: int = 2
) -> None:
    """Plot matrix heatmap.

    Args:
        matrix_values (np.array): Matrix data.
        title (str): The title of plot.
        x_labels (List[Union[int, str]]): The labels of x axis.
        y_labels (List[Union[int, str]]): The labels of y axis.
        x_title (str): The x axis title of plot.
        y_title (str): The y axis title of plot.
        save_path (str, optional): The file path you want to save the heatmap plot to. Defaults to None.
    """
    # change each element of z to type string for annotations
    matrix_values_text = [
        ['' if np.isnan(y) else f'{y: .{num_of_digits}f}' for y in x]
        for x in matrix_values
    ]

    # set up figure
    fig = ff.create_annotated_heatmap(
        matrix_values, x=x_labels, y=y_labels, font_colors=['black'],
        annotation_text=matrix_values_text, colorscale=colorscale
    )

    # add title
    fig.update_layout(title_text=title)

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text=x_title,
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.2,
                            y=0.5,
                            showarrow=False,
                            text=y_title,
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    fig.update_xaxes(tickangle=-45)
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    if save_path:
        fig.write_image(save_path)
    fig.show(show_mode)


def plot_pie(values: List[float], labels: List[str], return_fig: bool = False) -> Union[None, plotly.graph_objs._figure.Figure]:
    """Plot a pie cahrt.

    Args:
        values (List[float]): The list of values.
        labels (List[str]): All unique labels of `values`.
        return_fig (bool, optional): If True, return Plotly Figure object. Defaults to False.

    Returns:
        Union[None, plotly.graph_objs._figure.Figure]: Plotly figure object with pie chart.
    """
    fig = go.Figure(
        data=[
            go.Pie(
                values=values, labels=labels,
                textinfo='label+percent', textposition='inside'
            )
        ]
    )
    if return_fig:
        return fig
    fig.show(show_mode)


def plot_pies(
    list_of_values: List[List[float]], list_of_labels: List[List[str]], titles: List[str],
    col_num: int = 4, return_fig: bool = False, save_path: str = None
):

    row_num = math.ceil(len(list_of_values)/col_num)
    fig = make_subplots(
        rows=row_num, cols=col_num,
        specs=[[{'type': 'domain'}] * col_num] * row_num,
        subplot_titles=titles,
        vertical_spacing=0.03,
        horizontal_spacing=0.005
    )
    for idx, (labels, values) in enumerate(
        zip(
            list_of_labels, list_of_values
        )
    ):
        # Define pie charts
        fig.add_trace(
            go.Pie(
                labels=labels, values=values,
                name='Starry Night', textinfo='label+value+percent', textposition='inside'
            ), idx // col_num + 1, idx % col_num + 1
        )

    # Tune layout and hover info
    fig.update_layout(
        title_text="<i><b>Count error predicion classes in classes<b><i>",
        width=1000,
        height=1150,
    )
    fig = go.Figure(fig)
    if save_path:
        fig.write_image(save_path)
    fig.show(show_mode)


def plot_pred_img_and_pie(
    img_path: str, true_label: Union[str, int], pred_label: Union[str, int],
    label_probabilities: List[float], labels: List[str]
) -> None:
    """Plot a raw image with ground truth label and predict label in left plot, and plot a pie chart with predict probabilities in right plot.

    Args:
        img_path (str): Image file path.
        true_label (Union[str, int]): Image ground truth label.
        pred_label (Union[str, int]): Image predict label.
        label_probabilities (List[float]): The predict probabilities of labels.
        labels (List[str]): All unique labels.
    """
    title = f"<b><i>filepath: ...{img_path[-50:]}<br>true: {true_label} / predict: {pred_label}<i><b>"

    fig_img = plot_img(img_path, '', return_fig=True)
    fig_pie = plot_pie(label_probabilities, labels, return_fig=True)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "Image"}, {"type": "pie"}]],
        subplot_titles=['Raw image', "Predict probability"]
    )

    fig.add_trace(fig_img.data[0],
                  row=1, col=1)
    fig.add_trace(fig_pie.data[0],
                  row=1, col=2)
    fig.update_layout(title_text=title)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.show(show_mode)


def plot_pred_error_classes(
    y_true: List[Union[int, str]], y_pred: List[Union[int, str]], labels: List[str],
    col_num: int = 4, save_path: str = None
):
    """Calculate counts predict of label in each label and plot into pie charts.

    Args:
        y_true (List[Union[int, str]]): The list of ground truth labels.
        y_pred (List[Union[int, str]]): The list of predict labels.
        labels (List[Union[int, str]]): All unique labels.
        col_num (int, optional): The column number of subplots. Defaults to 4.
        save_path (str, optional): The file path you want to save the pie chars to. Defaults to None.
    """
    df = pd.DataFrame({'trueLabel': y_true, 'predLabel': y_pred})
    df['count'] = 1
    pred_error_df = df[df['trueLabel'].ne(df['predLabel'])]
    pred_error_count_df = pred_error_df.groupby(['trueLabel', 'predLabel']).count()[
        ['count']].reset_index()

    list_of_values = list()
    list_of_labels = list()
    titles = list()
    for label in labels:
        label_count_df = pred_error_count_df[pred_error_count_df['trueLabel'].eq(
            label)]
        label_count_df = label_count_df.sort_values('count', ascending=False)
        error_count = sum(label_count_df['count'])
        titles.append(f'{label}({error_count})')
        list_of_labels.append(label_count_df['predLabel'])
        list_of_values.append(label_count_df['count'])

    plot_pies(list_of_values, list_of_labels, titles,
              col_num=col_num, save_path=save_path)
