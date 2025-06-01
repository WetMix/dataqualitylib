# Databricks notebook source
import pandas as pd
from typing import List

from .building import Building
from .cug import CUG
from .measurement import Measurement

# COMMAND ----------

# MAGIC %run ./cug

# COMMAND ----------

# MAGIC %run ./building

# COMMAND ----------

# MAGIC %run ./measurement

# COMMAND ----------

class Input:
    """Ecoscada input representation"""
    def __init__(self, input_info: dict, measurement_info: dict, building_info: dict, cug_info: dict):
        self.__name = input_info['name']
        self.__gate_time_interval = input_info['gateTime']['interval']
        self.__gate_time_name = input_info['gateTime']['name']
        self.__medium_unit = input_info['medium']['primaryUnit']
        self.__type_id = input_info['inputType']
        self.__is_meter_bidirectional = input_info['isMeterBidirectional']
        self.__cug = CUG(cug_info)
        self.__building = Building(building_info)
        self.__measurements = [Measurement(measurement_info)]

    def link_to_additional_measurement(self, measurement_info: dict) -> None:
        self.__measurements.append(Measurement(measurement_info))

    def get_name(self) -> str:
        return self.__name

    def get_gate_time_interval_seconds(self) -> int:
        return self.__gate_time_interval

    def get_gate_time_interval_timedelta(self) -> pd.Timedelta:
        return pd.to_timedelta(self.__gate_time_interval, unit='s')

    def get_gate_time_name(self) -> str:
        return self.__gate_time_name

    def get_medium_unit(self) -> str:
        return self.__medium_unit

    def get_type_id(self) -> int:
        return self.__type_id

    @staticmethod
    def get_type_name_by_id(input_type_id: int) -> str:
        return {
            0: 'None',
            1: 'Pulse',
            2: 'Period',
            3: 'Gate',
            4: 'Alarm',
            5: 'MeterReading',
            6: 'Invoice'
        }[input_type_id]

    def is_meter_bidirectional(self) -> bool:
        return self.__is_meter_bidirectional

    def get_cug(self):
        return self.__cug

    def get_building(self):
        return self.__building

    def get_measurements(self) -> List[Measurement]:
        return self.__measurements