import os
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.cluster import DBSCAN
from azure.ai.anomalydetector import AnomalyDetectorClient
# from azure.ai.anomalydetector.models import TimeSeriesPoint
from azure.ai.anomalydetector.models import TimeSeriesPoint
from azure.ai.anomalydetector import AnomalyDetectorClient
from azure.ai.anomalydetector.models import UnivariateDetectionOptions
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials import AzureKeyCredential

from input import Input


class DQAnalyzer:
    """Implementation of data quality algorithms"""
    __ANOMALY_DETECTION_SENSIVITY = 0.5
    __MAX_FORECAST_DAYS = 14
    __ADMISSION_METERREADING_LIMIT = 0.05
    __SOLAR_DAYS_COUNT = 28
    __EPS = 10e-6

    __MSG_ISSUES_FOUND = 'Issues found'
    __MSG_ISSUES_NOT_FOUND = 'Issues not found'
    __MSG_CLASSIFICATION_DEFINED = 'Classification is defined'
    __MSG_CLASSIFICATION_NOT_DEFINED = 'Classification is not defined'

    # def __init__(self, algorithms: Tuple[bool]):
    #    self.__algorithms = algorithms

    # def analyze(self, df: pd.DataFrame) -> (pd.DataFrame, dict):
    #    general_report = dict()

    #    if self.__algorithms[0]:
    #        df, report = DQAnalyzer.__analyze_data_from_future(df)
    #        general_report['data_from_future'] = report
    #    if self.__algorithms[1]:
    #        df, report = DQAnalyzer.__analyze_gaps(df)
    #        general_report['gaps'] = report
    #    if self.__algorithms[2]:
    #        df, report = DQAnalyzer.__analyze_outliers(df)
    #        general_report['outliers'] = report
    #    if self.__algorithms[3]:
    #        report = DQAnalyzer.__analyze_anomalies(df)
    #        general_report['anomalies'] = report
    # if self.__algorithms[4]:
    #    report = DQAnalyzer.__classify_input_type(df, input_.is_meter_bidirectional(), input_.get_type_id())
    #    general_report['input_type'] = report
    # if self.__algorithms[5]:
    #    report = DQAnalyzer.__classify_input_profile(df, input_.get_gate_time_interval_timedelta())
    #    general_report['input_profile'] = report
    #   return df, general_report

    @staticmethod
    def analyze_data_from_future(df: pd.DataFrame) -> (pd.DataFrame, dict):  # работает
        '''
        Проверяет, нет ли записей во временном ряде с меткой времени, которая находится в будущем
        (по сравнению с текущим моментом). Это может быть признаком ошибки сбора данных.
        Возвращает:
            Количество точек из будущего.
            Границу времени (текущее время UTC).
            Статус: найдены/не найдены.
        '''
        boundary_timestamp = pd.Timestamp.now(tz='utc')
        data_from_future = df[df.index > boundary_timestamp]

        report = {
            'status': DQAnalyzer.__MSG_ISSUES_FOUND if len(data_from_future) > 0 else DQAnalyzer.__MSG_ISSUES_NOT_FOUND,
            'future_points_count': len(data_from_future),
            'boundary_timestamp': str(boundary_timestamp)
        }

        return df, report

    @staticmethod
    def analyze_gaps(df: pd.DataFrame) -> (pd.DataFrame, dict):  # работает
        '''
        Определяет наличие пропусков в данных по временному индексу. Если разница между соседними отметками
        времени больше 1 часа — это считается пропуском.
        Возвращает:
            Количество пропусков.
            Статус: есть/нет пропусков.
        '''
        df_sorted = df.sort_index()
        gaps_timestamps = df_sorted.index.to_series().diff().gt(pd.Timedelta("1h")).sum()

        report = {
            'status': DQAnalyzer.__MSG_ISSUES_FOUND if gaps_timestamps else DQAnalyzer.__MSG_ISSUES_NOT_FOUND,
            'gaps_count': int(gaps_timestamps)
        }

        return df, report

    @staticmethod
    def analyze_outliers(df: pd.DataFrame) -> (pd.DataFrame, dict):  # работает
        '''
        Обнаруживает выбросы на основе анализа разностей между соседними значениями.
        '''
        value_column = df.columns[0]

        absolute_differences = []
        for i in range(1, len(df)):
            absolute_differences.append(abs(df[value_column][i] - df[value_column][i - 1]))
        upper_quantile = pd.Series(absolute_differences).quantile(0.9)

        if abs(upper_quantile - 0.0) < DQAnalyzer.__EPS:
            differences_to_clustering = [[difference] for difference in
                                         absolute_differences[0: int(len(absolute_differences) / 10)]]
        else:
            differences_to_clustering = [[difference] for difference in absolute_differences if
                                         difference > upper_quantile]

        model = DBSCAN(eps=upper_quantile + DQAnalyzer.__EPS, min_samples=5).fit(differences_to_clustering)
        labels = list(model.labels_)

        acceptable_difference = 0
        for i in range(len(labels)):
            if labels[i] == 0 and acceptable_difference < differences_to_clustering[i][0]:
                acceptable_difference = differences_to_clustering[i][0]

        def is_outlier_triad(left_value, middle_value, right_value):
            lr_difference = abs(left_value - right_value)
            lm_difference = abs(left_value - middle_value)
            rm_difference = abs(right_value - middle_value)
            return not lr_difference > acceptable_difference \
                and lm_difference > acceptable_difference \
                and rm_difference > acceptable_difference

        outliers = []
        for i in range(1, len(df) - 1):
            if is_outlier_triad(df[value_column][i - 1], df[value_column][i], df[value_column][i + 1]):
                outliers.append({
                    'timestamp': str(df.index[i]),
                    'value': df[value_column][i]
                })
        report = {
            'status': DQAnalyzer.__MSG_ISSUES_FOUND if len(outliers) else DQAnalyzer.__MSG_ISSUES_NOT_FOUND,
            'outliers_count': len(outliers)
        }

        return df, report


@staticmethod
def analyze_anomalies(df: pd.DataFrame) -> dict:
    '''
        Использует внешний API Azure Anomaly Detector для определения аномальных точек во всём временном ряду.
        Возвращает:
            Список аномальных точек.
            Количество аномалий.
            Статус: найдены/не найдены.
     '''
    client = AnomalyDetectorClient(
            AzureKeyCredential(os.getenv('ANOMALY_DETECTOR_KEY')),
            os.getenv('ANOMALY_DETECTOR_ENDPOINT')
    )

    batch_size = 8640  # API limit for one request
    dataframe_batches = [df[i: i + batch_size] for i in range(0, len(df), batch_size)]

    anomalies = []

    for batch in dataframe_batches:
        #request = DetectRequest(
        #    series=[
        #        TimeSeriesPoint(timestamp=pd.Timestamp(index).isoformat(), value=items[0])
        #        for index, items in batch.iterrows()
        #    ],
        #    sensitivity=DQAnalyzer.__ANOMALY_DETECTION_SENSIVITY,
        #)
        request = UnivariateDetectionOptions(
            series=[
                TimeSeriesPoint(timestamp=pd.Timestamp(index).isoformat(), value=items[0])
                for index, items in batch.iterrows()
            ],
            sensitivity=DQAnalyzer._DQAnalyzer__ANOMALY_DETECTION_SENSIVITY,  # если переменная приватная
            granularity="daily"  # или "hourly" — укажи в зависимости от твоих данных
        )
        response = client.detect_entire_series(request)
        if any(response.is_anomaly):
            for i, value in enumerate(response.is_anomaly):
                if value:
                    anomalies.append({
                        'timestamp': str(batch.index[i]),
                        'value': batch[batch.columns[0]][i]
                    })

        report = {
            'status': DQAnalyzer.__MSG_ISSUES_FOUND if anomalies else DQAnalyzer.__MSG_ISSUES_NOT_FOUND,
            'anomalies_count': len(anomalies),
            'anomalies': anomalies
        }

        return report


    @staticmethod
    def __classify_input_type(df: pd.DataFrame, is_bidirectional: bool, input_type_id: int) -> dict:  # Работает
        value_column = df.columns[0]

        def check_for_meterreading():
            if is_bidirectional:
                direction_changes_count = 0
                direction_is_increasing = df[value_column][0] <= df[value_column][1]
                for i in range(2, len(df)):
                    if direction_is_increasing:
                        if df[value_column][i - 1] > df[value_column][i]:
                            direction_is_increasing = not direction_is_increasing
                            direction_changes_count += 1
                    else:
                        if df[value_column][i - 1] < df[value_column][i]:
                            direction_is_increasing = not direction_is_increasing
                            direction_changes_count += 1

                if direction_changes_count / len(df) > DQAnalyzer.__ADMISSION_METERREADING_LIMIT:
                    return False

            else:
                declines_count = 0
                for i in range(1, len(df)):
                    if df[value_column][i - 1] > df[value_column][i]:
                        declines_count += 1

                if (declines_count / len(df)) > DQAnalyzer.__ADMISSION_METERREADING_LIMIT:
                    return False

            return True

        def check_for_pulse():
            raise NotImplementedError

        is_meterreading = check_for_meterreading()
        is_pulse = not is_meterreading

        report = {
            'status':
                DQAnalyzer.__MSG_CLASSIFICATION_DEFINED if is_meterreading != is_pulse
                else DQAnalyzer.__MSG_CLASSIFICATION_NOT_DEFINED,
            'ecoscada_info': {
                'ecoscada_input_type': Input.get_type_name_by_id(input_type_id),
                'is_meter_bidirectional': is_bidirectional
            },
            'type_classification': {
                'is_meterreading': is_meterreading,
                'is_pulse': is_pulse
            }
        }

        return report

    @staticmethod
    def classify_input_profile(df: pd.DataFrame, timedelta: pd.Timedelta) -> dict:
        value_column = df.columns[0]

        def scaled_mean_absolute_error(y_true: pd.DataFrame, y_pred: pd.DataFrame):
            result = (y_true - y_pred).abs().sum() * np.mean(y_true.abs()) / len(y_true)
            if np.isnan(result[value_column]):
                return 100
            return result[value_column]

        def check_for_solar():
            freq = int(pd.Timedelta(1, unit='d') / timedelta)
            test_set = pd.read_csv(f'./solar_sample.csv', index_col=0)
            test_set = test_set / test_set.max()
            test_set = test_set.replace(np.inf, 0).fillna(0)
            df_period = df[:DQAnalyzer.__SOLAR_DAYS_COUNT * freq]
            df_period.index = [
                index.strftime('%m-%d %H:%M:%S')
                for index in df_period.index
            ]
            df_period = df_period / df_period.max()

            intersected_rows = test_set.loc[test_set.index.intersection(df_period.index)]
            df_test = pd.DataFrame({value_column: intersected_rows.iloc[:, 0]})

            solar_error = scaled_mean_absolute_error(df_test, df_period)
            for i in range(1, len(test_set.columns)):
                df_test = pd.DataFrame({value_column: intersected_rows.iloc[:, i]})
                error = scaled_mean_absolute_error(df_test, df_period)
                if error < solar_error:
                    solar_error = error

            return True if solar_error < 5 else False

        def check_for_grid():
            raise NotImplementedError

        is_solar = check_for_solar()
        is_grid = not is_solar

        report = {
            'status':
                DQAnalyzer.__MSG_CLASSIFICATION_DEFINED if is_solar != is_grid
                else DQAnalyzer.__MSG_CLASSIFICATION_NOT_DEFINED,
            'profile_classification': {
                'is_solar': is_solar,
                'is_grid': is_grid
            }
        }

        return report