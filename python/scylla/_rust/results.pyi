import ipaddress
from datetime import date, datetime, time
from decimal import Decimal
from typing import Dict, List, Union, Tuple, Any, Set
from uuid import UUID
from dateutil.relativedelta import relativedelta

ResultCqlNative = Union[
    # CQL:
    # - Counter
    # - TinyInt
    # - SmallInt
    # - Int
    # - BigInt
    # - Varint
    int,
    # CQL:
    # - Float
    # - Double
    float,
    # CQL:
    # - Ascii
    # - Text
    str,
    # CQL:
    # - Boolean
    bool,
    # CQL:
    # - Blob
    bytes,
    # CQL:
    # - Decimal
    Decimal,
    # CQL:
    # - Uuid
    # - Timeuuid
    UUID,
    # CQL:
    # - Inet (IPv4)
    ipaddress.IPv4Address,
    # CQL:
    # - Inet (IPv6)
    ipaddress.IPv6Address,
    # CQL:
    # - Date
    date,
    # CQL:
    # - Timestamp
    datetime,
    # CQL:
    # - Time
    time,
    # CQL:
    # - Duration
    relativedelta,
    # CQL:
    # - Empty
    # - null
    None,
]

ResultCqlCollection = Union[
    # CQL:
    # - List
    # - Vector
    List["ResultCqlValue"],
    # CQL:
    # - Set
    Set["ResultCqlValue"],
    # CQL:
    # - Tuple
    Tuple["ResultCqlValue", ...],
    # CQL:
    # - Map
    # - UserDefinedType (UDT)
    Dict["ResultCqlValue", "ResultCqlValue"],
]

ResultCqlValue = Union[
    ResultCqlNative,
    ResultCqlCollection,
]

class ColumnIterator:
    def __iter__(self) -> ColumnIterator: ...
    def __next__(self) -> Column: ...

class RowFactory:
    def __init__(self) -> None: ...
    def build(self, column_iterator: ColumnIterator) -> Dict[str, ResultCqlValue]: ...

class Column:
    @property
    def column_name(self) -> str: ...
    @property
    def value(self) -> ResultCqlValue: ...

class RowsIterator:
    def __next__(self) -> Any: ...
    def __iter__(self) -> RowsIterator: ...

class RequestResult:
    def iter_rows(self, factory: RowFactory | None = None) -> RowsIterator: ...
