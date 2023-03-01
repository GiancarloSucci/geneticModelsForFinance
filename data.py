from dataclasses import dataclass
from enum import Enum
from datetime import date
import nasdaqdatalink

class EDatasetType(Enum):
    '''Defines dataset type with Nasdaq Data Link code'''
    AAPL = 'WIKI/AAPL'

class ETimeInterval(Enum):
    '''Defines sample fequency with nasdaq strings'''
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    ANNUAL = 'annual'

@dataclass
class TimeSeriesConditions:
    start_date:date
    end_date:date
    time_interval:ETimeInterval

@dataclass
class TimeSeries:
    time_series_list:list
    conditions:TimeSeriesConditions
    

def get_nasdaq_time_series(dataset_type:EDatasetType, conditions:TimeSeriesConditions) -> TimeSeries:
    data_list = nasdaqdatalink.get(dataset_type.value,
                        start_date=conditions.start_date.strftime('%Y-%m-%d'),
                        end_date=conditions.end_date.strftime('%Y-%m-%d'),
                        collapse=conditions.time_interval.value)
    return TimeSeries(**{'time_series_list':data_list, 'conditions':conditions})

conditions = TimeSeriesConditions(start_date=date(2022, 1, 1),
                                   end_date=date(2023, 1, 1),
                                   time_interval=ETimeInterval.MONTHLY)
print(nasdaqdatalink.get('EIA/PET_RWTC_D', returns='numpy'))