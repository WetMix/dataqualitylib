# Databricks notebook source
class Measurement:
    """Ecoscada measurement representation"""
    def __init__(self, measurement_info: dict):
        self.__name = measurement_info['name']
        self.__guid = measurement_info['guid']
        self.__is_forecast = Measurement.__check_forecast_property(measurement_info)

    @staticmethod
    def __check_forecast_property(measurement_info: dict) -> bool:
        is_forecast = False
        for measurement_property in measurement_info['properties']:
            if measurement_property['code'] == 'IsForecast' and measurement_property['value'] == 'true':
                is_forecast = True
        return is_forecast

    def get_name(self):
        return self.__name

    def get_guid(self):
        return self.__guid

    def is_forecast(self):
        return self.__is_forecast