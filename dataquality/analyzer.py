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
import os
from .input import Input


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
    def analyze_data_from_future(df: pd.DataFrame, datetime_col: str) -> (pd.DataFrame, dict):
        """
        Проверяет, нет ли записей во временном ряде с меткой времени, которая находится в будущем
        (по сравнению с текущим моментом). Это может быть признаком ошибки сбора данных.
        Возвращает:
            Количество точек из будущего.
            Границу времени (текущее время UTC).
            Статус: найдены/не найдены.
        """
        df_copy = df.copy()

        # Преобразуем столбец в datetime с автоматическим определением формата
        df_copy[datetime_col] = pd.to_datetime(
            df_copy[datetime_col],
            errors='coerce',
            infer_datetime_format=True
        )

        # Проверяем успешность преобразования
        if df_copy[datetime_col].isnull().all():
            raise ValueError(f"Не удалось преобразовать столбец {datetime_col} в дату/время")

        # Устанавливаем индекс и сортируем
        df_copy = df_copy.set_index(datetime_col).sort_index()

        # Приводим индекс к UTC, если он не задан
        if df_copy.index.tz is None:
            df_copy = df_copy.tz_localize('UTC')

        # Получаем текущее время в UTC
        boundary_timestamp = pd.Timestamp.now(tz="utc")

        # Теперь сравнение будет корректным
        data_from_future = df_copy[df_copy.index > boundary_timestamp]

        report = {
            "status": (
                DQAnalyzer.__MSG_ISSUES_FOUND
                if len(data_from_future) > 0
                else DQAnalyzer.__MSG_ISSUES_NOT_FOUND
            ),
            "future_points_count": len(data_from_future),
            "boundary_timestamp": str(boundary_timestamp),
        }

        return df, report

    @staticmethod
    def analyze_gaps(df: pd.DataFrame, datetime_col: str) -> Tuple[pd.DataFrame, dict]:
        '''
        Определяет наличие пропусков в данных по временному индексу.
        Если разница между соседними отметками времени больше 1 часа - это считается пропуском.
        '''
        df_copy = df.copy()

        # Преобразуем столбец в datetime с автоматическим определением формата
        df_copy[datetime_col] = pd.to_datetime(
            df_copy[datetime_col],
            errors='coerce',
            infer_datetime_format=True
        )

        # Проверяем успешность преобразования
        if df_copy[datetime_col].isnull().all():
            raise ValueError(f"Не удалось преобразовать столбец {datetime_col} в дату/время")

        # Устанавливаем индекс и сортируем
        df_copy = df_copy.set_index(datetime_col).sort_index()

        # Вычисляем разницу между соседними значениями
        time_diff = df_copy.index.to_series().diff()
        gaps_count = time_diff.gt(pd.Timedelta("1h")).sum()

        report = {
            'status': DQAnalyzer.__MSG_ISSUES_FOUND if gaps_count else DQAnalyzer.__MSG_ISSUES_NOT_FOUND,
            'gaps_count': int(gaps_count)
        }

        return df, report

    @staticmethod
    def analyze_outliers(df: pd.DataFrame) -> (pd.DataFrame, dict):
        """
        Обнаруживает выбросы на основе анализа разностей между соседними значениями.
        """
        if len(df) < 2:
            return df, {"status": "Недостаточно данных", "outliers_count": 0}

        value_column = df.columns[0]

        absolute_differences = []
        for i in range(1, len(df)):
            absolute_differences.append(
                abs(df[value_column].iloc[i] - df[value_column].iloc[i - 1])
            )

        if not absolute_differences:
            return df, {"status": "Нет данных для анализа", "outliers_count": 0}

        upper_quantile = pd.Series(absolute_differences).quantile(0.9)

        if abs(upper_quantile - 0.0) < DQAnalyzer.__EPS:
            differences_to_clustering = [
                [difference]
                for difference in absolute_differences[
                                  0: int(len(absolute_differences) / 10)
                                  ]
            ]
        else:
            differences_to_clustering = [
                [difference]
                for difference in absolute_differences
                if difference > upper_quantile
            ]

        if not differences_to_clustering:
            return df, {"status": "Нет данных для кластеризации", "outliers_count": 0}

        model = DBSCAN(eps=upper_quantile + DQAnalyzer.__EPS, min_samples=5).fit(
            differences_to_clustering
        )
        labels = list(model.labels_)

        acceptable_difference = 0
        for i in range(len(labels)):
            if (
                    labels[i] == 0
                    and acceptable_difference < differences_to_clustering[i][0]
            ):
                acceptable_difference = differences_to_clustering[i][0]

        def is_outlier_triad(left_value, middle_value, right_value):
            lr_difference = abs(left_value - right_value)
            lm_difference = abs(left_value - middle_value)
            rm_difference = abs(right_value - middle_value)
            return (
                    not lr_difference > acceptable_difference
                    and lm_difference > acceptable_difference
                    and rm_difference > acceptable_difference
            )

        outliers = []
        for i in range(1, len(df) - 1):
            if is_outlier_triad(
                    df[value_column][i - 1], df[value_column][i], df[value_column][i + 1]
            ):
                outliers.append(
                    {"timestamp": str(df.index[i]), "value": df[value_column][i]}
                )

        report = {
            "status": (
                DQAnalyzer.__MSG_ISSUES_FOUND
                if len(outliers)
                else DQAnalyzer.__MSG_ISSUES_NOT_FOUND
            ),
            "outliers_count": len(outliers),
            "outliers": outliers,
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

           return df, report

    @staticmethod
    def classify_input_type(
            df: pd.DataFrame, is_bidirectional: bool, input_type_id: int
    ) -> dict:

        df = df.select_dtypes(['int64', 'float64'])
        value_column = df.columns[0]

        # Преобразуем значения в числовой тип (float)
        df_copy = df.copy()
        df_copy[value_column] = pd.to_numeric(df_copy[value_column], errors='coerce')

        # Проверяем, есть ли непреобразуемые значения
        if df_copy[value_column].isnull().all():
            raise ValueError(f"Не удалось преобразовать колонку '{value_column}' в числовой тип")

        some_threshold = 10
        some_pulse_limit = 0.2

        def check_for_meterreading():
            if is_bidirectional:
                direction_changes_count = 0
                direction_is_increasing = (
                        df_copy[value_column].iloc[0] <= df_copy[value_column].iloc[1]
                )
                for i in range(2, len(df_copy)):
                    if direction_is_increasing:
                        if df_copy[value_column].iloc[i - 1] > df_copy[value_column].iloc[i]:
                            direction_is_increasing = not direction_is_increasing
                            direction_changes_count += 1
                    else:
                        if df_copy[value_column].iloc[i - 1] < df_copy[value_column].iloc[i]:
                            direction_is_increasing = not direction_is_increasing
                            direction_changes_count += 1

                if direction_changes_count / len(df_copy) > DQAnalyzer.__ADMISSION_METERREADING_LIMIT:
                    return False

            else:
                declines_count = 0
                for i in range(1, len(df_copy)):
                    if df_copy[value_column].iloc[i - 1] > df_copy[value_column].iloc[i]:
                        declines_count += 1

                if declines_count / len(df_copy) > DQAnalyzer.__ADMISSION_METERREADING_LIMIT:
                    return False

            return True

        def check_for_pulse():
            pulse_count = 0
            for i in range(1, len(df_copy)):
                if abs(df_copy[value_column].iloc[i] - df_copy[value_column].iloc[i - 1]) > some_threshold:
                    pulse_count += 1

            if pulse_count / len(df_copy) > some_pulse_limit:
                return True
            return False

        is_meterreading = check_for_meterreading()
        is_pulse = check_for_pulse()

        report = {
            "status": (
                DQAnalyzer.__MSG_CLASSIFICATION_DEFINED
                if is_meterreading != is_pulse
                else DQAnalyzer.__MSG_CLASSIFICATION_NOT_DEFINED
            ),
            "ecoscada_info": {
                "ecoscada_input_type": Input.get_type_name_by_id(input_type_id),
                "is_meter_bidirectional": is_bidirectional,
            },
            "type_classification": {
                "is_meterreading": is_meterreading,
                "is_pulse": is_pulse,
            },
        }

        return report
    #Пока что не работает


    @staticmethod
    def classify_input_profile(df: pd.DataFrame, datetime_col: str, path: str, timedelta: pd.Timedelta) -> dict:
       df_copy = df.copy()

       # 1. Преобразуем столбец в datetime
       try:
           df_copy[datetime_col] = pd.to_datetime(
               df_copy[datetime_col],
               errors='coerce',
               infer_datetime_format=True
           )
           if df_copy[datetime_col].isnull().all():
               raise ValueError(f"Column {datetime_col} cannot be converted to datetime")
       except Exception as e:
           raise ValueError(f"Datetime conversion failed: {str(e)}")

    # 2. Устанавливаем datetime как индекс
       df_copy = df_copy.set_index(datetime_col)

    # 3. Проверяем solar профиль
       def scaled_mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
           result = (y_true - y_pred).abs().mean() * y_true.abs().mean()
           return 100 if np.isnan(result) else result

       def check_for_solar() -> bool:
           try:
               freq = int(pd.Timedelta(days=1) / timedelta)  # Закрывающая скобка была пропущена
               test_set = pd.read_csv(path, index_col=0)
               test_set = test_set / test_set.max()
               test_set = test_set.replace([np.inf, -np.inf], 0).fillna(0)

            # Берем первые N дней данных
               df_period = df_copy.iloc[:int(DQAnalyzer.__SOLAR_DAYS_COUNT * freq)]

            # Нормализуем данные
               df_period_normalized = df_period / df_period.max()

            # Сравниваем с тестовыми данными
               solar_error = float('inf')
               for test_col in test_set.columns:
                   common_index = test_set.index.intersection(df_period_normalized.index)
                   if len(common_index) == 0:
                       continue

                   current_error = scaled_mean_absolute_error(
                       test_set.loc[common_index, test_col],
                       df_period_normalized.loc[common_index, df_period_normalized.columns[0]]
                   )

                   if current_error < solar_error:
                       solar_error = current_error

               return solar_error < 5  # Пороговое значение

           except Exception as e:
               print(f"Solar check error: {str(e)}")
               return False

       is_solar = check_for_solar()
       is_grid = not is_solar

       return {
           'status': (
               DQAnalyzer.__MSG_CLASSIFICATION_DEFINED
               if is_solar != is_grid
               else DQAnalyzer.__MSG_CLASSIFICATION_NOT_DEFINED
           ),
           'profile_classification': {
               'is_solar': is_solar,
               'is_grid': is_grid
           }
       }