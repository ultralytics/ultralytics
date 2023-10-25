from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DlRequest(_message.Message):
    __slots__ = ["type", "imdata"]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    IMDATA_FIELD_NUMBER: _ClassVar[int]
    type: int
    imdata: bytes
    def __init__(self, type: _Optional[int] = ..., imdata: _Optional[bytes] = ...) -> None: ...

class DlBoundingRect(_message.Message):
    __slots__ = ["x", "y", "w", "h"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    w: int
    h: int
    def __init__(self, x: _Optional[int] = ..., y: _Optional[int] = ..., w: _Optional[int] = ..., h: _Optional[int] = ...) -> None: ...

class DlPoint(_message.Message):
    __slots__ = ["x", "y"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    def __init__(self, x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...

class DlMask(_message.Message):
    __slots__ = ["points"]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    points: _containers.RepeatedCompositeFieldContainer[DlPoint]
    def __init__(self, points: _Optional[_Iterable[_Union[DlPoint, _Mapping]]] = ...) -> None: ...

class DlResult(_message.Message):
    __slots__ = ["classid", "score", "rect", "mask"]
    CLASSID_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    RECT_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    classid: str
    score: float
    rect: DlBoundingRect
    mask: DlMask
    def __init__(self, classid: _Optional[str] = ..., score: _Optional[float] = ..., rect: _Optional[_Union[DlBoundingRect, _Mapping]] = ..., mask: _Optional[_Union[DlMask, _Mapping]] = ...) -> None: ...

class DlResponse(_message.Message):
    __slots__ = ["results"]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[DlResult]
    def __init__(self, results: _Optional[_Iterable[_Union[DlResult, _Mapping]]] = ...) -> None: ...

class DlEmbeddingRequest(_message.Message):
    __slots__ = ["imdata", "imsize"]
    IMDATA_FIELD_NUMBER: _ClassVar[int]
    IMSIZE_FIELD_NUMBER: _ClassVar[int]
    imdata: bytes
    imsize: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, imdata: _Optional[bytes] = ..., imsize: _Optional[_Iterable[int]] = ...) -> None: ...

class Tensor(_message.Message):
    __slots__ = ["data", "shape"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, data: _Optional[_Iterable[float]] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...
