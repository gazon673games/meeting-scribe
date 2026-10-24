from __future__ import annotations

import ctypes


class GUID(ctypes.Structure):
    _fields_ = [
        ('Data1', ctypes.c_uint32),
        ('Data2', ctypes.c_uint16),
        ('Data3', ctypes.c_uint16),
        ('Data4', ctypes.c_uint8 * 8),
    ]


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [('fmtid', GUID), ('pid', ctypes.c_uint32)]


class PROPVARIANT_VALUE(ctypes.Union):
    _fields_ = [
        ('pwszVal', ctypes.c_wchar_p),
        ('raw', ctypes.c_byte * 16),
    ]


class PROPVARIANT(ctypes.Structure):
    _anonymous_ = ('value',)
    _fields_ = [
        ('vt', ctypes.c_ushort),
        ('wReserved1', ctypes.c_ushort),
        ('wReserved2', ctypes.c_ushort),
        ('wReserved3', ctypes.c_ushort),
        ('value', PROPVARIANT_VALUE),
    ]
