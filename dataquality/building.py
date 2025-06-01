# Databricks notebook source
class Building:
    """Ecoscada building representation"""
    def __init__(self, building_info: dict):
        self.__name = building_info['name']
        self.__guid = building_info['guid']

    def get_name(self):
        return self.__name

    def get_guid(self):
        return self.__guid