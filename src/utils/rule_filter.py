from typing import List

import pandas as pd


class RuleFilter:

    def __init__(self):
        self._labels = ['banana', 'bareland', 'carrot', 'corn', 'dragonfruit', 'garlic',
                        'guava', 'peanut', 'pineapple', 'pumpkin', 'rice', 'soybean', 'sugarcane', 'tomato']
        self._no_data_month_df = self._build_rule_df()
        self._no_data_cammodel_df = self._bulid_camrule_df()

    def _build_rule_df(self) -> pd.DataFrame:
        no_data_months = {
            'bareland': [11],
            'carrot': [3, 4, 5, 6, 7, 8],
            'corn': [11],
            'garlic': [6, 8],
            'peanut': [7, 11],
            'rice': [6, 7, 8, 9, 10, 11, 12],
        }
        no_data_month_df = pd.DataFrame(
            {'label': no_data_months.keys(), 'month': no_data_months.values()})
        no_data_month_df = no_data_month_df.explode('month')
        no_data_month_df = no_data_month_df.groupby(
            'month')['label'].apply(list).to_frame()
        return no_data_month_df

    def _bulid_camrule_df(self):

        no_data_months = {'COOLPIX AW120  ': [],
                          'FinePix F200EXR': [],
                          'NIKON D90': [],
                          'DMC-GF3': ['corn', 'rice'],
                          'COOLPIX P330': [],
                          'DMC-LX5': ['carrot', 'peanut', 'rice', 'garlic', 'corn'],
                          'DMC-GF2': ['carrot', 'rice'],
                          'TG-610          ':  ['bareland', 'pumpkin', 'pineapple', 'carrot', 'tomato', 'soybean', 'sugarcane', 'peanut', 'guava', 'rice', 'garlic', 'dragonfruit', 'corn'],
                          'DSC-P150': ['bareland', 'pumpkin', 'pineapple', 'carrot', 'banana', 'tomato', 'soybean', 'sugarcane', 'peanut', 'guava', 'rice', 'garlic', 'dragonfruit'],
                          'Canon DIGITAL IXUS 200 IS': ['garlic', 'pumpkin', 'carrot', 'banana', 'tomato', 'sugarcane', 'guava', 'rice', 'pineapple', 'dragonfruit', 'corn'],
                          '': ['banana', 'bareland']}

        no_data_month_df = pd.DataFrame(
            {'cammodel': no_data_months.keys(), 'label': no_data_months.values()})
        no_data_month_df = no_data_month_df.explode('label')
        no_data_month_df = no_data_month_df.groupby(
            'label')['cammodel'].apply(list).to_frame()
        return no_data_month_df

    def get_pred_label(self, pred_probs: List[float], month: int, cam_model: str) -> str:
        label_probs = dict(zip(self._labels, pred_probs))
        pred_labels = sorted(label_probs, key=label_probs.get, reverse=True)
        if month in self._no_data_month_df.index:
            pred_labels = [
                label for label in pred_labels
                if label not in self._no_data_month_df['label'].loc[month]
            ]
        pred_labels = [
            label for label in pred_labels if cam_model not in self._no_data_cammodel_df.loc[label]['cammodel']]
        pred_label = pred_labels[0]
        return pred_label


if __name__ == "__main__":
    rule_filter = RuleFilter()
    pred_probs = [0.5, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(rule_filter.get_pred_label(pred_probs, 1, 'COOLPIX P330'))
    print(rule_filter.get_pred_label(pred_probs, 1, 'DSC-P150'))
    # print(rule_filter.get_pred_label(pred_probs, 3))  # banana
    # print(rule_filter.get_pred_label(pred_probs, 1))  # carrot

    pred_probs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.5]
    print(rule_filter.get_pred_label(pred_probs, 11))  # tomato
    print(rule_filter.get_pred_label(pred_probs, 2))  # rice
