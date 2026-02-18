from typing import Sequence, TypeAlias, Mapping, Union, Tuple, AbstractSet
import ipaddress
from datetime import date, datetime, time
from dateutil.relativedelta import relativedelta
from decimal import Decimal
from uuid import UUID

class UnsetType:
    """
    Type of the Unset singleton.
    """

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

RequestCqlNative = Union[
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

RequestCqlCollection = Union[
    # CQL:
    # - List
    # - Vector
    Sequence["RequestCqlValue"],
    # CQL:
    # - Set
    AbstractSet["RequestCqlValue"],
    # CQL:
    # - Tuple
    Tuple["RequestCqlValue", ...],
    # CQL:
    # - Map
    # - UserDefinedType (UDT)
    Mapping["RequestCqlValue", "RequestCqlValue"],
]

RequestCqlValue = Union[
    RequestCqlNative,
    RequestCqlCollection,
]


CqlValueList: TypeAlias = Sequence[RequestCqlValue] | Mapping[str, RequestCqlValue]
