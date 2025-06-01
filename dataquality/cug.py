# Databricks notebook source
class CUG:
    """Ecoscada CUG representation"""
    def __init__(self, cug_info: dict):
        self.__name = cug_info['name']
        self.__guid = cug_info['guid']

    def get_name(self):
        return self.__name

    def get_guid(self):
        return self.__guid