from ctypes import *

isl = cdll.LoadLibrary("libisl.so")
libc = cdll.LoadLibrary("libc.so.6")

class Error(Exception):
    pass

class Context:
    defaultInstance = None

    def __init__(self):
        ptr = isl.isl_ctx_alloc()
        self.ptr = ptr

    def __del__(self):
        isl.isl_ctx_free(self)

    def from_param(self):
        return self.ptr

    @staticmethod
    def getDefaultInstance():
        if Context.defaultInstance == None:
            Context.defaultInstance = Context()
        return Context.defaultInstance

isl.isl_ctx_alloc.restype = c_void_p
isl.isl_ctx_free.argtypes = [Context]

class union_pw_multi_aff(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_multi_aff_read_from_str(self.ctx, args[0])
            return
        if len(args) == 1 and args[0].__class__ is union_pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_multi_aff_from_union_pw_aff(isl.isl_union_pw_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is pw_multi_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_multi_aff_from_pw_multi_aff(isl.isl_pw_multi_aff_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_pw_multi_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        ptr = isl.isl_union_pw_multi_aff_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_pw_multi_aff("""%s""")' % s
        else:
            return 'isl.union_pw_multi_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_multi_aff:
                arg1 = union_pw_multi_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_add(isl.isl_union_pw_multi_aff_copy(arg0.ptr), isl.isl_union_pw_multi_aff_copy(arg1.ptr))
        return union_pw_multi_aff(ctx=ctx, ptr=res)
    def flat_range_product(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_multi_aff:
                arg1 = union_pw_multi_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_flat_range_product(isl.isl_union_pw_multi_aff_copy(arg0.ptr), isl.isl_union_pw_multi_aff_copy(arg1.ptr))
        return union_pw_multi_aff(ctx=ctx, ptr=res)
    def pullback(arg0, arg1):
        if arg1.__class__ is union_pw_multi_aff:
            res = isl.isl_union_pw_multi_aff_pullback_union_pw_multi_aff(isl.isl_union_pw_multi_aff_copy(arg0.ptr), isl.isl_union_pw_multi_aff_copy(arg1.ptr))
            return union_pw_multi_aff(ctx=arg0.ctx, ptr=res)
    def union_add(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_multi_aff:
                arg0 = union_pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_multi_aff:
                arg1 = union_pw_multi_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_pw_multi_aff_union_add(isl.isl_union_pw_multi_aff_copy(arg0.ptr), isl.isl_union_pw_multi_aff_copy(arg1.ptr))
        return union_pw_multi_aff(ctx=ctx, ptr=res)

isl.isl_union_pw_multi_aff_read_from_str.restype = c_void_p
isl.isl_union_pw_multi_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_union_pw_multi_aff_from_union_pw_aff.restype = c_void_p
isl.isl_union_pw_multi_aff_from_union_pw_aff.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_from_pw_multi_aff.restype = c_void_p
isl.isl_union_pw_multi_aff_from_pw_multi_aff.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_add.restype = c_void_p
isl.isl_union_pw_multi_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_flat_range_product.restype = c_void_p
isl.isl_union_pw_multi_aff_flat_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_pullback_union_pw_multi_aff.restype = c_void_p
isl.isl_union_pw_multi_aff_pullback_union_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_union_add.restype = c_void_p
isl.isl_union_pw_multi_aff_union_add.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_multi_aff_free.restype = c_void_p
isl.isl_union_pw_multi_aff_free.argtypes = [c_void_p]
isl.isl_union_pw_multi_aff_to_str.restype = POINTER(c_char)
isl.isl_union_pw_multi_aff_to_str.argtypes = [c_void_p]

class multi_union_pw_aff(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is union_pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_union_pw_aff_from_union_pw_aff(isl.isl_union_pw_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is multi_pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_union_pw_aff_from_multi_pw_aff(isl.isl_multi_pw_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_union_pw_aff_read_from_str(self.ctx, args[0])
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_multi_union_pw_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        ptr = isl.isl_multi_union_pw_aff_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.multi_union_pw_aff("""%s""")' % s
        else:
            return 'isl.multi_union_pw_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_union_pw_aff:
                arg1 = multi_union_pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_add(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_multi_union_pw_aff_copy(arg1.ptr))
        return multi_union_pw_aff(ctx=ctx, ptr=res)
    def flat_range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_union_pw_aff:
                arg1 = multi_union_pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_flat_range_product(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_multi_union_pw_aff_copy(arg1.ptr))
        return multi_union_pw_aff(ctx=ctx, ptr=res)
    def pullback(arg0, arg1):
        if arg1.__class__ is union_pw_multi_aff:
            res = isl.isl_multi_union_pw_aff_pullback_union_pw_multi_aff(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_union_pw_multi_aff_copy(arg1.ptr))
            return multi_union_pw_aff(ctx=arg0.ctx, ptr=res)
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_union_pw_aff:
                arg1 = multi_union_pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_range_product(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_multi_union_pw_aff_copy(arg1.ptr))
        return multi_union_pw_aff(ctx=ctx, ptr=res)
    def union_add(arg0, arg1):
        try:
            if not arg0.__class__ is multi_union_pw_aff:
                arg0 = multi_union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_union_pw_aff:
                arg1 = multi_union_pw_aff(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_union_pw_aff_union_add(isl.isl_multi_union_pw_aff_copy(arg0.ptr), isl.isl_multi_union_pw_aff_copy(arg1.ptr))
        return multi_union_pw_aff(ctx=ctx, ptr=res)

isl.isl_multi_union_pw_aff_from_union_pw_aff.restype = c_void_p
isl.isl_multi_union_pw_aff_from_union_pw_aff.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_from_multi_pw_aff.restype = c_void_p
isl.isl_multi_union_pw_aff_from_multi_pw_aff.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_read_from_str.restype = c_void_p
isl.isl_multi_union_pw_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_multi_union_pw_aff_add.restype = c_void_p
isl.isl_multi_union_pw_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_flat_range_product.restype = c_void_p
isl.isl_multi_union_pw_aff_flat_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_pullback_union_pw_multi_aff.restype = c_void_p
isl.isl_multi_union_pw_aff_pullback_union_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_range_product.restype = c_void_p
isl.isl_multi_union_pw_aff_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_union_add.restype = c_void_p
isl.isl_multi_union_pw_aff_union_add.argtypes = [c_void_p, c_void_p]
isl.isl_multi_union_pw_aff_free.restype = c_void_p
isl.isl_multi_union_pw_aff_free.argtypes = [c_void_p]
isl.isl_multi_union_pw_aff_to_str.restype = POINTER(c_char)
isl.isl_multi_union_pw_aff_to_str.argtypes = [c_void_p]

class union_pw_aff(union_pw_multi_aff, multi_union_pw_aff):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_aff_from_pw_aff(isl.isl_pw_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_pw_aff_read_from_str(self.ctx, args[0])
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_pw_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        ptr = isl.isl_union_pw_aff_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_pw_aff("""%s""")' % s
        else:
            return 'isl.union_pw_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_aff:
                arg1 = union_pw_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).add(arg1)
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_add(isl.isl_union_pw_aff_copy(arg0.ptr), isl.isl_union_pw_aff_copy(arg1.ptr))
        return union_pw_aff(ctx=ctx, ptr=res)
    def pullback(arg0, arg1):
        if arg1.__class__ is union_pw_multi_aff:
            res = isl.isl_union_pw_aff_pullback_union_pw_multi_aff(isl.isl_union_pw_aff_copy(arg0.ptr), isl.isl_union_pw_multi_aff_copy(arg1.ptr))
            return union_pw_aff(ctx=arg0.ctx, ptr=res)
    def union_add(arg0, arg1):
        try:
            if not arg0.__class__ is union_pw_aff:
                arg0 = union_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_pw_aff:
                arg1 = union_pw_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).union_add(arg1)
        ctx = arg0.ctx
        res = isl.isl_union_pw_aff_union_add(isl.isl_union_pw_aff_copy(arg0.ptr), isl.isl_union_pw_aff_copy(arg1.ptr))
        return union_pw_aff(ctx=ctx, ptr=res)

isl.isl_union_pw_aff_from_pw_aff.restype = c_void_p
isl.isl_union_pw_aff_from_pw_aff.argtypes = [c_void_p]
isl.isl_union_pw_aff_read_from_str.restype = c_void_p
isl.isl_union_pw_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_union_pw_aff_add.restype = c_void_p
isl.isl_union_pw_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_pullback_union_pw_multi_aff.restype = c_void_p
isl.isl_union_pw_aff_pullback_union_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_union_add.restype = c_void_p
isl.isl_union_pw_aff_union_add.argtypes = [c_void_p, c_void_p]
isl.isl_union_pw_aff_free.restype = c_void_p
isl.isl_union_pw_aff_free.argtypes = [c_void_p]
isl.isl_union_pw_aff_to_str.restype = POINTER(c_char)
isl.isl_union_pw_aff_to_str.argtypes = [c_void_p]

class multi_pw_aff(multi_union_pw_aff):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is multi_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_pw_aff_from_multi_aff(isl.isl_multi_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_pw_aff_from_pw_aff(isl.isl_pw_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is pw_multi_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_pw_aff_from_pw_multi_aff(isl.isl_pw_multi_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_pw_aff_read_from_str(self.ctx, args[0])
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_multi_pw_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        ptr = isl.isl_multi_pw_aff_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.multi_pw_aff("""%s""")' % s
        else:
            return 'isl.multi_pw_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_pw_aff:
                arg1 = multi_pw_aff(arg1)
        except:
            return multi_union_pw_aff(arg0).add(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_add(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
        return multi_pw_aff(ctx=ctx, ptr=res)
    def flat_range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_pw_aff:
                arg1 = multi_pw_aff(arg1)
        except:
            return multi_union_pw_aff(arg0).flat_range_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_flat_range_product(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
        return multi_pw_aff(ctx=ctx, ptr=res)
    def product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_pw_aff:
                arg1 = multi_pw_aff(arg1)
        except:
            return multi_union_pw_aff(arg0).product(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_product(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
        return multi_pw_aff(ctx=ctx, ptr=res)
    def pullback(arg0, arg1):
        if arg1.__class__ is multi_aff:
            res = isl.isl_multi_pw_aff_pullback_multi_aff(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
            return multi_pw_aff(ctx=arg0.ctx, ptr=res)
        if arg1.__class__ is pw_multi_aff:
            res = isl.isl_multi_pw_aff_pullback_pw_multi_aff(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
            return multi_pw_aff(ctx=arg0.ctx, ptr=res)
        if arg1.__class__ is multi_pw_aff:
            res = isl.isl_multi_pw_aff_pullback_multi_pw_aff(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
            return multi_pw_aff(ctx=arg0.ctx, ptr=res)
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_pw_aff:
                arg0 = multi_pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_pw_aff:
                arg1 = multi_pw_aff(arg1)
        except:
            return multi_union_pw_aff(arg0).range_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_pw_aff_range_product(isl.isl_multi_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
        return multi_pw_aff(ctx=ctx, ptr=res)

isl.isl_multi_pw_aff_from_multi_aff.restype = c_void_p
isl.isl_multi_pw_aff_from_multi_aff.argtypes = [c_void_p]
isl.isl_multi_pw_aff_from_pw_aff.restype = c_void_p
isl.isl_multi_pw_aff_from_pw_aff.argtypes = [c_void_p]
isl.isl_multi_pw_aff_from_pw_multi_aff.restype = c_void_p
isl.isl_multi_pw_aff_from_pw_multi_aff.argtypes = [c_void_p]
isl.isl_multi_pw_aff_read_from_str.restype = c_void_p
isl.isl_multi_pw_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_multi_pw_aff_add.restype = c_void_p
isl.isl_multi_pw_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_flat_range_product.restype = c_void_p
isl.isl_multi_pw_aff_flat_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_product.restype = c_void_p
isl.isl_multi_pw_aff_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_pullback_multi_aff.restype = c_void_p
isl.isl_multi_pw_aff_pullback_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_pullback_pw_multi_aff.restype = c_void_p
isl.isl_multi_pw_aff_pullback_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_pullback_multi_pw_aff.restype = c_void_p
isl.isl_multi_pw_aff_pullback_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_range_product.restype = c_void_p
isl.isl_multi_pw_aff_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_pw_aff_free.restype = c_void_p
isl.isl_multi_pw_aff_free.argtypes = [c_void_p]
isl.isl_multi_pw_aff_to_str.restype = POINTER(c_char)
isl.isl_multi_pw_aff_to_str.argtypes = [c_void_p]

class pw_multi_aff(union_pw_multi_aff, multi_pw_aff):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is multi_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_multi_aff_from_multi_aff(isl.isl_multi_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is pw_aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_multi_aff_from_pw_aff(isl.isl_pw_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_multi_aff_read_from_str(self.ctx, args[0])
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_pw_multi_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        ptr = isl.isl_pw_multi_aff_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.pw_multi_aff("""%s""")' % s
        else:
            return 'isl.pw_multi_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_multi_aff:
                arg1 = pw_multi_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).add(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_add(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
        return pw_multi_aff(ctx=ctx, ptr=res)
    def flat_range_product(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_multi_aff:
                arg1 = pw_multi_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).flat_range_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_flat_range_product(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
        return pw_multi_aff(ctx=ctx, ptr=res)
    def product(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_multi_aff:
                arg1 = pw_multi_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).product(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_product(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
        return pw_multi_aff(ctx=ctx, ptr=res)
    def pullback(arg0, arg1):
        if arg1.__class__ is multi_aff:
            res = isl.isl_pw_multi_aff_pullback_multi_aff(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
            return pw_multi_aff(ctx=arg0.ctx, ptr=res)
        if arg1.__class__ is pw_multi_aff:
            res = isl.isl_pw_multi_aff_pullback_pw_multi_aff(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
            return pw_multi_aff(ctx=arg0.ctx, ptr=res)
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_multi_aff:
                arg1 = pw_multi_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).range_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_range_product(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
        return pw_multi_aff(ctx=ctx, ptr=res)
    def union_add(arg0, arg1):
        try:
            if not arg0.__class__ is pw_multi_aff:
                arg0 = pw_multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_multi_aff:
                arg1 = pw_multi_aff(arg1)
        except:
            return union_pw_multi_aff(arg0).union_add(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_multi_aff_union_add(isl.isl_pw_multi_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
        return pw_multi_aff(ctx=ctx, ptr=res)

isl.isl_pw_multi_aff_from_multi_aff.restype = c_void_p
isl.isl_pw_multi_aff_from_multi_aff.argtypes = [c_void_p]
isl.isl_pw_multi_aff_from_pw_aff.restype = c_void_p
isl.isl_pw_multi_aff_from_pw_aff.argtypes = [c_void_p]
isl.isl_pw_multi_aff_read_from_str.restype = c_void_p
isl.isl_pw_multi_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_pw_multi_aff_add.restype = c_void_p
isl.isl_pw_multi_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_flat_range_product.restype = c_void_p
isl.isl_pw_multi_aff_flat_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_product.restype = c_void_p
isl.isl_pw_multi_aff_product.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_pullback_multi_aff.restype = c_void_p
isl.isl_pw_multi_aff_pullback_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_pullback_pw_multi_aff.restype = c_void_p
isl.isl_pw_multi_aff_pullback_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_range_product.restype = c_void_p
isl.isl_pw_multi_aff_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_union_add.restype = c_void_p
isl.isl_pw_multi_aff_union_add.argtypes = [c_void_p, c_void_p]
isl.isl_pw_multi_aff_free.restype = c_void_p
isl.isl_pw_multi_aff_free.argtypes = [c_void_p]
isl.isl_pw_multi_aff_to_str.restype = POINTER(c_char)
isl.isl_pw_multi_aff_to_str.argtypes = [c_void_p]

class pw_aff(union_pw_aff, pw_multi_aff, multi_pw_aff):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_aff_from_aff(isl.isl_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_pw_aff_read_from_str(self.ctx, args[0])
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_pw_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        ptr = isl.isl_pw_aff_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.pw_aff("""%s""")' % s
        else:
            return 'isl.pw_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).add(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_add(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        return pw_aff(ctx=ctx, ptr=res)
    def pullback(arg0, arg1):
        if arg1.__class__ is multi_aff:
            res = isl.isl_pw_aff_pullback_multi_aff(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
            return pw_aff(ctx=arg0.ctx, ptr=res)
        if arg1.__class__ is pw_multi_aff:
            res = isl.isl_pw_aff_pullback_pw_multi_aff(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_multi_aff_copy(arg1.ptr))
            return pw_aff(ctx=arg0.ctx, ptr=res)
        if arg1.__class__ is multi_pw_aff:
            res = isl.isl_pw_aff_pullback_multi_pw_aff(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_multi_pw_aff_copy(arg1.ptr))
            return pw_aff(ctx=arg0.ctx, ptr=res)
    def union_add(arg0, arg1):
        try:
            if not arg0.__class__ is pw_aff:
                arg0 = pw_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is pw_aff:
                arg1 = pw_aff(arg1)
        except:
            return union_pw_aff(arg0).union_add(arg1)
        ctx = arg0.ctx
        res = isl.isl_pw_aff_union_add(isl.isl_pw_aff_copy(arg0.ptr), isl.isl_pw_aff_copy(arg1.ptr))
        return pw_aff(ctx=ctx, ptr=res)

isl.isl_pw_aff_from_aff.restype = c_void_p
isl.isl_pw_aff_from_aff.argtypes = [c_void_p]
isl.isl_pw_aff_read_from_str.restype = c_void_p
isl.isl_pw_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_pw_aff_add.restype = c_void_p
isl.isl_pw_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_pullback_multi_aff.restype = c_void_p
isl.isl_pw_aff_pullback_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_pullback_pw_multi_aff.restype = c_void_p
isl.isl_pw_aff_pullback_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_pullback_multi_pw_aff.restype = c_void_p
isl.isl_pw_aff_pullback_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_union_add.restype = c_void_p
isl.isl_pw_aff_union_add.argtypes = [c_void_p, c_void_p]
isl.isl_pw_aff_free.restype = c_void_p
isl.isl_pw_aff_free.argtypes = [c_void_p]
isl.isl_pw_aff_to_str.restype = POINTER(c_char)
isl.isl_pw_aff_to_str.argtypes = [c_void_p]

class multi_aff(pw_multi_aff, multi_pw_aff):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is aff:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_aff_from_aff(isl.isl_aff_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_multi_aff_read_from_str(self.ctx, args[0])
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_multi_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        ptr = isl.isl_multi_aff_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.multi_aff("""%s""")' % s
        else:
            return 'isl.multi_aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_aff:
                arg1 = multi_aff(arg1)
        except:
            return pw_multi_aff(arg0).add(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_add(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
        return multi_aff(ctx=ctx, ptr=res)
    def flat_range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_aff:
                arg1 = multi_aff(arg1)
        except:
            return pw_multi_aff(arg0).flat_range_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_flat_range_product(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
        return multi_aff(ctx=ctx, ptr=res)
    def product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_aff:
                arg1 = multi_aff(arg1)
        except:
            return pw_multi_aff(arg0).product(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_product(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
        return multi_aff(ctx=ctx, ptr=res)
    def pullback(arg0, arg1):
        if arg1.__class__ is multi_aff:
            res = isl.isl_multi_aff_pullback_multi_aff(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
            return multi_aff(ctx=arg0.ctx, ptr=res)
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_aff:
                arg0 = multi_aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_aff:
                arg1 = multi_aff(arg1)
        except:
            return pw_multi_aff(arg0).range_product(arg1)
        ctx = arg0.ctx
        res = isl.isl_multi_aff_range_product(isl.isl_multi_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
        return multi_aff(ctx=ctx, ptr=res)

isl.isl_multi_aff_from_aff.restype = c_void_p
isl.isl_multi_aff_from_aff.argtypes = [c_void_p]
isl.isl_multi_aff_read_from_str.restype = c_void_p
isl.isl_multi_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_multi_aff_add.restype = c_void_p
isl.isl_multi_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_flat_range_product.restype = c_void_p
isl.isl_multi_aff_flat_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_product.restype = c_void_p
isl.isl_multi_aff_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_pullback_multi_aff.restype = c_void_p
isl.isl_multi_aff_pullback_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_range_product.restype = c_void_p
isl.isl_multi_aff_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_aff_free.restype = c_void_p
isl.isl_multi_aff_free.argtypes = [c_void_p]
isl.isl_multi_aff_to_str.restype = POINTER(c_char)
isl.isl_multi_aff_to_str.argtypes = [c_void_p]

class aff(pw_aff, multi_aff):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_aff_read_from_str(self.ctx, args[0])
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_aff_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        ptr = isl.isl_aff_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.aff("""%s""")' % s
        else:
            return 'isl.aff("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is aff:
                arg0 = aff(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return pw_aff(arg0).add(arg1)
        ctx = arg0.ctx
        res = isl.isl_aff_add(isl.isl_aff_copy(arg0.ptr), isl.isl_aff_copy(arg1.ptr))
        return aff(ctx=ctx, ptr=res)
    def pullback(arg0, arg1):
        if arg1.__class__ is multi_aff:
            res = isl.isl_aff_pullback_multi_aff(isl.isl_aff_copy(arg0.ptr), isl.isl_multi_aff_copy(arg1.ptr))
            return aff(ctx=arg0.ctx, ptr=res)

isl.isl_aff_read_from_str.restype = c_void_p
isl.isl_aff_read_from_str.argtypes = [Context, c_char_p]
isl.isl_aff_add.restype = c_void_p
isl.isl_aff_add.argtypes = [c_void_p, c_void_p]
isl.isl_aff_pullback_multi_aff.restype = c_void_p
isl.isl_aff_pullback_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_aff_free.restype = c_void_p
isl.isl_aff_free.argtypes = [c_void_p]
isl.isl_aff_to_str.restype = POINTER(c_char)
isl.isl_aff_to_str.argtypes = [c_void_p]

class ast_build(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 0:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_ast_build_alloc(self.ctx)
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_build_free(self.ptr)
    def access_from(arg0, arg1):
        if arg1.__class__ is pw_multi_aff:
            res = isl.isl_ast_build_access_from_pw_multi_aff(arg0.ptr, isl.isl_pw_multi_aff_copy(arg1.ptr))
            return ast_expr(ctx=arg0.ctx, ptr=res)
        if arg1.__class__ is multi_pw_aff:
            res = isl.isl_ast_build_access_from_multi_pw_aff(arg0.ptr, isl.isl_multi_pw_aff_copy(arg1.ptr))
            return ast_expr(ctx=arg0.ctx, ptr=res)
    def call_from(arg0, arg1):
        if arg1.__class__ is pw_multi_aff:
            res = isl.isl_ast_build_call_from_pw_multi_aff(arg0.ptr, isl.isl_pw_multi_aff_copy(arg1.ptr))
            return ast_expr(ctx=arg0.ctx, ptr=res)
        if arg1.__class__ is multi_pw_aff:
            res = isl.isl_ast_build_call_from_multi_pw_aff(arg0.ptr, isl.isl_multi_pw_aff_copy(arg1.ptr))
            return ast_expr(ctx=arg0.ctx, ptr=res)
    def expr_from(arg0, arg1):
        if arg1.__class__ is set:
            res = isl.isl_ast_build_expr_from_set(arg0.ptr, isl.isl_set_copy(arg1.ptr))
            return ast_expr(ctx=arg0.ctx, ptr=res)
        if arg1.__class__ is pw_aff:
            res = isl.isl_ast_build_expr_from_pw_aff(arg0.ptr, isl.isl_pw_aff_copy(arg1.ptr))
            return ast_expr(ctx=arg0.ctx, ptr=res)
    @staticmethod
    def from_context(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_build_from_context(isl.isl_set_copy(arg0.ptr))
        return ast_build(ctx=ctx, ptr=res)
    def node_from_schedule_map(arg0, arg1):
        try:
            if not arg0.__class__ is ast_build:
                arg0 = ast_build(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_build_node_from_schedule_map(arg0.ptr, isl.isl_union_map_copy(arg1.ptr))
        return ast_node(ctx=ctx, ptr=res)

isl.isl_ast_build_alloc.restype = c_void_p
isl.isl_ast_build_alloc.argtypes = [Context]
isl.isl_ast_build_access_from_pw_multi_aff.restype = c_void_p
isl.isl_ast_build_access_from_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_access_from_multi_pw_aff.restype = c_void_p
isl.isl_ast_build_access_from_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_call_from_pw_multi_aff.restype = c_void_p
isl.isl_ast_build_call_from_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_call_from_multi_pw_aff.restype = c_void_p
isl.isl_ast_build_call_from_multi_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_expr_from_set.restype = c_void_p
isl.isl_ast_build_expr_from_set.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_expr_from_pw_aff.restype = c_void_p
isl.isl_ast_build_expr_from_pw_aff.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_from_context.restype = c_void_p
isl.isl_ast_build_from_context.argtypes = [c_void_p]
isl.isl_ast_build_node_from_schedule_map.restype = c_void_p
isl.isl_ast_build_node_from_schedule_map.argtypes = [c_void_p, c_void_p]
isl.isl_ast_build_free.restype = c_void_p
isl.isl_ast_build_free.argtypes = [c_void_p]

class ast_expr(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_expr_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_expr:
                arg0 = ast_expr(arg0)
        except:
            raise
        ptr = isl.isl_ast_expr_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_expr("""%s""")' % s
        else:
            return 'isl.ast_expr("%s")' % s
    def to_C_str(arg0):
        try:
            if not arg0.__class__ is ast_expr:
                arg0 = ast_expr(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_expr_to_C_str(arg0.ptr)
        if res == 0:
            raise
        string = str(cast(res, c_char_p).value)
        libc.free(res)
        return string

isl.isl_ast_expr_to_C_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_C_str.argtypes = [c_void_p]
isl.isl_ast_expr_free.restype = c_void_p
isl.isl_ast_expr_free.argtypes = [c_void_p]
isl.isl_ast_expr_to_str.restype = POINTER(c_char)
isl.isl_ast_expr_to_str.argtypes = [c_void_p]

class ast_node(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_ast_node_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ptr = isl.isl_ast_node_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.ast_node("""%s""")' % s
        else:
            return 'isl.ast_node("%s")' % s
    def to_C_str(arg0):
        try:
            if not arg0.__class__ is ast_node:
                arg0 = ast_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_ast_node_to_C_str(arg0.ptr)
        if res == 0:
            raise
        string = str(cast(res, c_char_p).value)
        libc.free(res)
        return string

isl.isl_ast_node_to_C_str.restype = POINTER(c_char)
isl.isl_ast_node_to_C_str.argtypes = [c_void_p]
isl.isl_ast_node_free.restype = c_void_p
isl.isl_ast_node_free.argtypes = [c_void_p]
isl.isl_ast_node_to_str.restype = POINTER(c_char)
isl.isl_ast_node_to_str.argtypes = [c_void_p]

class union_map(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is basic_map:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_map_from_basic_map(isl.isl_basic_map_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is map:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_map_from_map(isl.isl_map_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_map_read_from_str(self.ctx, args[0])
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_map_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ptr = isl.isl_union_map_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_map("""%s""")' % s
        else:
            return 'isl.union_map("%s")' % s
    def affine_hull(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_affine_hull(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def apply_domain(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_apply_domain(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def apply_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_apply_range(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def coalesce(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_coalesce(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def compute_divs(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_compute_divs(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def deltas(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_deltas(isl.isl_union_map_copy(arg0.ptr))
        return union_set(ctx=ctx, ptr=res)
    def detect_equalities(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_detect_equalities(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def domain(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_domain(isl.isl_union_map_copy(arg0.ptr))
        return union_set(ctx=ctx, ptr=res)
    def domain_factor_domain(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_domain_factor_domain(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def domain_factor_range(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_domain_factor_range(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def domain_map(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_domain_map(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def domain_map_union_pw_multi_aff(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_domain_map_union_pw_multi_aff(isl.isl_union_map_copy(arg0.ptr))
        return union_pw_multi_aff(ctx=ctx, ptr=res)
    def domain_product(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_domain_product(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def factor_domain(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_factor_domain(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def factor_range(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_factor_range(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def fixed_power(arg0, arg1):
        if arg1.__class__ is val:
            res = isl.isl_union_map_fixed_power_val(isl.isl_union_map_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
            return union_map(ctx=arg0.ctx, ptr=res)
    def foreach_map(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = map(ctx=arg0.ctx, ptr=cb_arg0)
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_union_map_foreach_map(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        return res
    @staticmethod
    def convert_from(arg0):
        if arg0.__class__ is union_pw_multi_aff:
            res = isl.isl_union_map_from_union_pw_multi_aff(isl.isl_union_pw_multi_aff_copy(arg0.ptr))
            return union_map(ctx=arg0.ctx, ptr=res)
        if arg0.__class__ is multi_union_pw_aff:
            res = isl.isl_union_map_from_multi_union_pw_aff(isl.isl_multi_union_pw_aff_copy(arg0.ptr))
            return union_map(ctx=arg0.ctx, ptr=res)
    @staticmethod
    def from_domain_and_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_from_domain_and_range(isl.isl_union_set_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_gist(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def gist_domain(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_gist_domain(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def gist_params(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_gist_params(isl.isl_union_map_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def gist_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_gist_range(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def intersect(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_intersect(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def intersect_domain(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_intersect_domain(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_intersect_params(isl.isl_union_map_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def intersect_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_intersect_range(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def is_bijective(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_bijective(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_empty(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_empty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_injective(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_injective(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_single_valued(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_single_valued(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_strict_subset(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_strict_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_subset(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_is_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def lexmax(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_lexmax(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def lexmin(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_lexmin(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def polyhedral_hull(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_polyhedral_hull(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def product(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_product(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def range(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_range(isl.isl_union_map_copy(arg0.ptr))
        return union_set(ctx=ctx, ptr=res)
    def range_factor_domain(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_range_factor_domain(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def range_factor_range(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_range_factor_range(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def range_map(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_range_map(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_range_product(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def reverse(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_reverse(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def subtract(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_subtract(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def subtract_domain(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_subtract_domain(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def subtract_range(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_subtract_range(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def union(arg0, arg1):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_union(isl.isl_union_map_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        return union_map(ctx=ctx, ptr=res)
    def wrap(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_wrap(isl.isl_union_map_copy(arg0.ptr))
        return union_set(ctx=ctx, ptr=res)
    def zip(arg0):
        try:
            if not arg0.__class__ is union_map:
                arg0 = union_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_map_zip(isl.isl_union_map_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)

isl.isl_union_map_from_basic_map.restype = c_void_p
isl.isl_union_map_from_basic_map.argtypes = [c_void_p]
isl.isl_union_map_from_map.restype = c_void_p
isl.isl_union_map_from_map.argtypes = [c_void_p]
isl.isl_union_map_read_from_str.restype = c_void_p
isl.isl_union_map_read_from_str.argtypes = [Context, c_char_p]
isl.isl_union_map_affine_hull.restype = c_void_p
isl.isl_union_map_affine_hull.argtypes = [c_void_p]
isl.isl_union_map_apply_domain.restype = c_void_p
isl.isl_union_map_apply_domain.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_apply_range.restype = c_void_p
isl.isl_union_map_apply_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_coalesce.restype = c_void_p
isl.isl_union_map_coalesce.argtypes = [c_void_p]
isl.isl_union_map_compute_divs.restype = c_void_p
isl.isl_union_map_compute_divs.argtypes = [c_void_p]
isl.isl_union_map_deltas.restype = c_void_p
isl.isl_union_map_deltas.argtypes = [c_void_p]
isl.isl_union_map_detect_equalities.restype = c_void_p
isl.isl_union_map_detect_equalities.argtypes = [c_void_p]
isl.isl_union_map_domain.restype = c_void_p
isl.isl_union_map_domain.argtypes = [c_void_p]
isl.isl_union_map_domain_factor_domain.restype = c_void_p
isl.isl_union_map_domain_factor_domain.argtypes = [c_void_p]
isl.isl_union_map_domain_factor_range.restype = c_void_p
isl.isl_union_map_domain_factor_range.argtypes = [c_void_p]
isl.isl_union_map_domain_map.restype = c_void_p
isl.isl_union_map_domain_map.argtypes = [c_void_p]
isl.isl_union_map_domain_map_union_pw_multi_aff.restype = c_void_p
isl.isl_union_map_domain_map_union_pw_multi_aff.argtypes = [c_void_p]
isl.isl_union_map_domain_product.restype = c_void_p
isl.isl_union_map_domain_product.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_factor_domain.restype = c_void_p
isl.isl_union_map_factor_domain.argtypes = [c_void_p]
isl.isl_union_map_factor_range.restype = c_void_p
isl.isl_union_map_factor_range.argtypes = [c_void_p]
isl.isl_union_map_fixed_power_val.restype = c_void_p
isl.isl_union_map_fixed_power_val.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_foreach_map.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_union_map_from_union_pw_multi_aff.restype = c_void_p
isl.isl_union_map_from_union_pw_multi_aff.argtypes = [c_void_p]
isl.isl_union_map_from_multi_union_pw_aff.restype = c_void_p
isl.isl_union_map_from_multi_union_pw_aff.argtypes = [c_void_p]
isl.isl_union_map_from_domain_and_range.restype = c_void_p
isl.isl_union_map_from_domain_and_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_gist.restype = c_void_p
isl.isl_union_map_gist.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_gist_domain.restype = c_void_p
isl.isl_union_map_gist_domain.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_gist_params.restype = c_void_p
isl.isl_union_map_gist_params.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_gist_range.restype = c_void_p
isl.isl_union_map_gist_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect.restype = c_void_p
isl.isl_union_map_intersect.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect_domain.restype = c_void_p
isl.isl_union_map_intersect_domain.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect_params.restype = c_void_p
isl.isl_union_map_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_intersect_range.restype = c_void_p
isl.isl_union_map_intersect_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_is_bijective.restype = c_bool
isl.isl_union_map_is_bijective.argtypes = [c_void_p]
isl.isl_union_map_is_empty.restype = c_bool
isl.isl_union_map_is_empty.argtypes = [c_void_p]
isl.isl_union_map_is_equal.restype = c_bool
isl.isl_union_map_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_is_injective.restype = c_bool
isl.isl_union_map_is_injective.argtypes = [c_void_p]
isl.isl_union_map_is_single_valued.restype = c_bool
isl.isl_union_map_is_single_valued.argtypes = [c_void_p]
isl.isl_union_map_is_strict_subset.restype = c_bool
isl.isl_union_map_is_strict_subset.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_is_subset.restype = c_bool
isl.isl_union_map_is_subset.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_lexmax.restype = c_void_p
isl.isl_union_map_lexmax.argtypes = [c_void_p]
isl.isl_union_map_lexmin.restype = c_void_p
isl.isl_union_map_lexmin.argtypes = [c_void_p]
isl.isl_union_map_polyhedral_hull.restype = c_void_p
isl.isl_union_map_polyhedral_hull.argtypes = [c_void_p]
isl.isl_union_map_product.restype = c_void_p
isl.isl_union_map_product.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_range.restype = c_void_p
isl.isl_union_map_range.argtypes = [c_void_p]
isl.isl_union_map_range_factor_domain.restype = c_void_p
isl.isl_union_map_range_factor_domain.argtypes = [c_void_p]
isl.isl_union_map_range_factor_range.restype = c_void_p
isl.isl_union_map_range_factor_range.argtypes = [c_void_p]
isl.isl_union_map_range_map.restype = c_void_p
isl.isl_union_map_range_map.argtypes = [c_void_p]
isl.isl_union_map_range_product.restype = c_void_p
isl.isl_union_map_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_reverse.restype = c_void_p
isl.isl_union_map_reverse.argtypes = [c_void_p]
isl.isl_union_map_subtract.restype = c_void_p
isl.isl_union_map_subtract.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_subtract_domain.restype = c_void_p
isl.isl_union_map_subtract_domain.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_subtract_range.restype = c_void_p
isl.isl_union_map_subtract_range.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_union.restype = c_void_p
isl.isl_union_map_union.argtypes = [c_void_p, c_void_p]
isl.isl_union_map_wrap.restype = c_void_p
isl.isl_union_map_wrap.argtypes = [c_void_p]
isl.isl_union_map_zip.restype = c_void_p
isl.isl_union_map_zip.argtypes = [c_void_p]
isl.isl_union_map_free.restype = c_void_p
isl.isl_union_map_free.argtypes = [c_void_p]
isl.isl_union_map_to_str.restype = POINTER(c_char)
isl.isl_union_map_to_str.argtypes = [c_void_p]

class map(union_map):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_map_read_from_str(self.ctx, args[0])
            return
        if len(args) == 1 and args[0].__class__ is basic_map:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_map_from_basic_map(isl.isl_basic_map_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_map_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ptr = isl.isl_map_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.map("""%s""")' % s
        else:
            return 'isl.map("%s")' % s
    def affine_hull(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_affine_hull(isl.isl_map_copy(arg0.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def apply_domain(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).apply_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_apply_domain(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        return map(ctx=ctx, ptr=res)
    def apply_range(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).apply_range(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_apply_range(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        return map(ctx=ctx, ptr=res)
    def coalesce(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_coalesce(isl.isl_map_copy(arg0.ptr))
        return map(ctx=ctx, ptr=res)
    def complement(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_complement(isl.isl_map_copy(arg0.ptr))
        return map(ctx=ctx, ptr=res)
    def deltas(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_deltas(isl.isl_map_copy(arg0.ptr))
        return set(ctx=ctx, ptr=res)
    def detect_equalities(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_detect_equalities(isl.isl_map_copy(arg0.ptr))
        return map(ctx=ctx, ptr=res)
    def flatten(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_flatten(isl.isl_map_copy(arg0.ptr))
        return map(ctx=ctx, ptr=res)
    def flatten_domain(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_flatten_domain(isl.isl_map_copy(arg0.ptr))
        return map(ctx=ctx, ptr=res)
    def flatten_range(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_flatten_range(isl.isl_map_copy(arg0.ptr))
        return map(ctx=ctx, ptr=res)
    def foreach_basic_map(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = basic_map(ctx=arg0.ctx, ptr=cb_arg0)
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_map_foreach_basic_map(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        return res
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_gist(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        return map(ctx=ctx, ptr=res)
    def gist_domain(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_map(arg0).gist_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_gist_domain(isl.isl_map_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        return map(ctx=ctx, ptr=res)
    def intersect(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).intersect(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_intersect(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        return map(ctx=ctx, ptr=res)
    def intersect_domain(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_map(arg0).intersect_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_intersect_domain(isl.isl_map_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        return map(ctx=ctx, ptr=res)
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_map(arg0).intersect_params(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_intersect_params(isl.isl_map_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        return map(ctx=ctx, ptr=res)
    def intersect_range(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_map(arg0).intersect_range(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_intersect_range(isl.isl_map_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        return map(ctx=ctx, ptr=res)
    def is_bijective(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_is_bijective(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_disjoint(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).is_disjoint(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_is_disjoint(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_empty(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_is_empty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).is_equal(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_injective(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_is_injective(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_single_valued(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_is_single_valued(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_strict_subset(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).is_strict_subset(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_is_strict_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_subset(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).is_subset(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_is_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def lexmax(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_lexmax(isl.isl_map_copy(arg0.ptr))
        return map(ctx=ctx, ptr=res)
    def lexmin(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_lexmin(isl.isl_map_copy(arg0.ptr))
        return map(ctx=ctx, ptr=res)
    def polyhedral_hull(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_polyhedral_hull(isl.isl_map_copy(arg0.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def reverse(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_reverse(isl.isl_map_copy(arg0.ptr))
        return map(ctx=ctx, ptr=res)
    def sample(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_sample(isl.isl_map_copy(arg0.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def subtract(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).subtract(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_subtract(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        return map(ctx=ctx, ptr=res)
    def union(arg0, arg1):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_map(arg0).union(arg1)
        ctx = arg0.ctx
        res = isl.isl_map_union(isl.isl_map_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        return map(ctx=ctx, ptr=res)
    def unshifted_simple_hull(arg0):
        try:
            if not arg0.__class__ is map:
                arg0 = map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_map_unshifted_simple_hull(isl.isl_map_copy(arg0.ptr))
        return basic_map(ctx=ctx, ptr=res)

isl.isl_map_read_from_str.restype = c_void_p
isl.isl_map_read_from_str.argtypes = [Context, c_char_p]
isl.isl_map_from_basic_map.restype = c_void_p
isl.isl_map_from_basic_map.argtypes = [c_void_p]
isl.isl_map_affine_hull.restype = c_void_p
isl.isl_map_affine_hull.argtypes = [c_void_p]
isl.isl_map_apply_domain.restype = c_void_p
isl.isl_map_apply_domain.argtypes = [c_void_p, c_void_p]
isl.isl_map_apply_range.restype = c_void_p
isl.isl_map_apply_range.argtypes = [c_void_p, c_void_p]
isl.isl_map_coalesce.restype = c_void_p
isl.isl_map_coalesce.argtypes = [c_void_p]
isl.isl_map_complement.restype = c_void_p
isl.isl_map_complement.argtypes = [c_void_p]
isl.isl_map_deltas.restype = c_void_p
isl.isl_map_deltas.argtypes = [c_void_p]
isl.isl_map_detect_equalities.restype = c_void_p
isl.isl_map_detect_equalities.argtypes = [c_void_p]
isl.isl_map_flatten.restype = c_void_p
isl.isl_map_flatten.argtypes = [c_void_p]
isl.isl_map_flatten_domain.restype = c_void_p
isl.isl_map_flatten_domain.argtypes = [c_void_p]
isl.isl_map_flatten_range.restype = c_void_p
isl.isl_map_flatten_range.argtypes = [c_void_p]
isl.isl_map_foreach_basic_map.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_map_gist.restype = c_void_p
isl.isl_map_gist.argtypes = [c_void_p, c_void_p]
isl.isl_map_gist_domain.restype = c_void_p
isl.isl_map_gist_domain.argtypes = [c_void_p, c_void_p]
isl.isl_map_intersect.restype = c_void_p
isl.isl_map_intersect.argtypes = [c_void_p, c_void_p]
isl.isl_map_intersect_domain.restype = c_void_p
isl.isl_map_intersect_domain.argtypes = [c_void_p, c_void_p]
isl.isl_map_intersect_params.restype = c_void_p
isl.isl_map_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_map_intersect_range.restype = c_void_p
isl.isl_map_intersect_range.argtypes = [c_void_p, c_void_p]
isl.isl_map_is_bijective.restype = c_bool
isl.isl_map_is_bijective.argtypes = [c_void_p]
isl.isl_map_is_disjoint.restype = c_bool
isl.isl_map_is_disjoint.argtypes = [c_void_p, c_void_p]
isl.isl_map_is_empty.restype = c_bool
isl.isl_map_is_empty.argtypes = [c_void_p]
isl.isl_map_is_equal.restype = c_bool
isl.isl_map_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_map_is_injective.restype = c_bool
isl.isl_map_is_injective.argtypes = [c_void_p]
isl.isl_map_is_single_valued.restype = c_bool
isl.isl_map_is_single_valued.argtypes = [c_void_p]
isl.isl_map_is_strict_subset.restype = c_bool
isl.isl_map_is_strict_subset.argtypes = [c_void_p, c_void_p]
isl.isl_map_is_subset.restype = c_bool
isl.isl_map_is_subset.argtypes = [c_void_p, c_void_p]
isl.isl_map_lexmax.restype = c_void_p
isl.isl_map_lexmax.argtypes = [c_void_p]
isl.isl_map_lexmin.restype = c_void_p
isl.isl_map_lexmin.argtypes = [c_void_p]
isl.isl_map_polyhedral_hull.restype = c_void_p
isl.isl_map_polyhedral_hull.argtypes = [c_void_p]
isl.isl_map_reverse.restype = c_void_p
isl.isl_map_reverse.argtypes = [c_void_p]
isl.isl_map_sample.restype = c_void_p
isl.isl_map_sample.argtypes = [c_void_p]
isl.isl_map_subtract.restype = c_void_p
isl.isl_map_subtract.argtypes = [c_void_p, c_void_p]
isl.isl_map_union.restype = c_void_p
isl.isl_map_union.argtypes = [c_void_p, c_void_p]
isl.isl_map_unshifted_simple_hull.restype = c_void_p
isl.isl_map_unshifted_simple_hull.argtypes = [c_void_p]
isl.isl_map_free.restype = c_void_p
isl.isl_map_free.argtypes = [c_void_p]
isl.isl_map_to_str.restype = POINTER(c_char)
isl.isl_map_to_str.argtypes = [c_void_p]

class basic_map(map):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_basic_map_read_from_str(self.ctx, args[0])
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_basic_map_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ptr = isl.isl_basic_map_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.basic_map("""%s""")' % s
        else:
            return 'isl.basic_map("%s")' % s
    def affine_hull(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_affine_hull(isl.isl_basic_map_copy(arg0.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def apply_domain(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).apply_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_apply_domain(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_map_copy(arg1.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def apply_range(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).apply_range(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_apply_range(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_map_copy(arg1.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def deltas(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_deltas(isl.isl_basic_map_copy(arg0.ptr))
        return basic_set(ctx=ctx, ptr=res)
    def detect_equalities(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_detect_equalities(isl.isl_basic_map_copy(arg0.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def flatten(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_flatten(isl.isl_basic_map_copy(arg0.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def flatten_domain(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_flatten_domain(isl.isl_basic_map_copy(arg0.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def flatten_range(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_flatten_range(isl.isl_basic_map_copy(arg0.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_gist(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_map_copy(arg1.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def intersect(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).intersect(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_intersect(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_map_copy(arg1.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def intersect_domain(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return map(arg0).intersect_domain(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_intersect_domain(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_set_copy(arg1.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def intersect_range(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return map(arg0).intersect_range(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_intersect_range(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_set_copy(arg1.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def is_empty(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_is_empty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).is_equal(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_subset(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).is_subset(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_is_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def lexmax(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_lexmax(isl.isl_basic_map_copy(arg0.ptr))
        return map(ctx=ctx, ptr=res)
    def lexmin(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_lexmin(isl.isl_basic_map_copy(arg0.ptr))
        return map(ctx=ctx, ptr=res)
    def reverse(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_reverse(isl.isl_basic_map_copy(arg0.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def sample(arg0):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_map_sample(isl.isl_basic_map_copy(arg0.ptr))
        return basic_map(ctx=ctx, ptr=res)
    def union(arg0, arg1):
        try:
            if not arg0.__class__ is basic_map:
                arg0 = basic_map(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return map(arg0).union(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_map_union(isl.isl_basic_map_copy(arg0.ptr), isl.isl_basic_map_copy(arg1.ptr))
        return map(ctx=ctx, ptr=res)

isl.isl_basic_map_read_from_str.restype = c_void_p
isl.isl_basic_map_read_from_str.argtypes = [Context, c_char_p]
isl.isl_basic_map_affine_hull.restype = c_void_p
isl.isl_basic_map_affine_hull.argtypes = [c_void_p]
isl.isl_basic_map_apply_domain.restype = c_void_p
isl.isl_basic_map_apply_domain.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_apply_range.restype = c_void_p
isl.isl_basic_map_apply_range.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_deltas.restype = c_void_p
isl.isl_basic_map_deltas.argtypes = [c_void_p]
isl.isl_basic_map_detect_equalities.restype = c_void_p
isl.isl_basic_map_detect_equalities.argtypes = [c_void_p]
isl.isl_basic_map_flatten.restype = c_void_p
isl.isl_basic_map_flatten.argtypes = [c_void_p]
isl.isl_basic_map_flatten_domain.restype = c_void_p
isl.isl_basic_map_flatten_domain.argtypes = [c_void_p]
isl.isl_basic_map_flatten_range.restype = c_void_p
isl.isl_basic_map_flatten_range.argtypes = [c_void_p]
isl.isl_basic_map_gist.restype = c_void_p
isl.isl_basic_map_gist.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_intersect.restype = c_void_p
isl.isl_basic_map_intersect.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_intersect_domain.restype = c_void_p
isl.isl_basic_map_intersect_domain.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_intersect_range.restype = c_void_p
isl.isl_basic_map_intersect_range.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_is_empty.restype = c_bool
isl.isl_basic_map_is_empty.argtypes = [c_void_p]
isl.isl_basic_map_is_equal.restype = c_bool
isl.isl_basic_map_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_is_subset.restype = c_bool
isl.isl_basic_map_is_subset.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_lexmax.restype = c_void_p
isl.isl_basic_map_lexmax.argtypes = [c_void_p]
isl.isl_basic_map_lexmin.restype = c_void_p
isl.isl_basic_map_lexmin.argtypes = [c_void_p]
isl.isl_basic_map_reverse.restype = c_void_p
isl.isl_basic_map_reverse.argtypes = [c_void_p]
isl.isl_basic_map_sample.restype = c_void_p
isl.isl_basic_map_sample.argtypes = [c_void_p]
isl.isl_basic_map_union.restype = c_void_p
isl.isl_basic_map_union.argtypes = [c_void_p, c_void_p]
isl.isl_basic_map_free.restype = c_void_p
isl.isl_basic_map_free.argtypes = [c_void_p]
isl.isl_basic_map_to_str.restype = POINTER(c_char)
isl.isl_basic_map_to_str.argtypes = [c_void_p]

class union_set(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is basic_set:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_set_from_basic_set(isl.isl_basic_set_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is set:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_set_from_set(isl.isl_set_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is point:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_set_from_point(isl.isl_point_copy(args[0].ptr))
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_set_read_from_str(self.ctx, args[0])
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_set_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ptr = isl.isl_union_set_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_set("""%s""")' % s
        else:
            return 'isl.union_set("%s")' % s
    def affine_hull(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_affine_hull(isl.isl_union_set_copy(arg0.ptr))
        return union_set(ctx=ctx, ptr=res)
    def apply(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_apply(isl.isl_union_set_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        return union_set(ctx=ctx, ptr=res)
    def coalesce(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_coalesce(isl.isl_union_set_copy(arg0.ptr))
        return union_set(ctx=ctx, ptr=res)
    def compute_divs(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_compute_divs(isl.isl_union_set_copy(arg0.ptr))
        return union_set(ctx=ctx, ptr=res)
    def detect_equalities(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_detect_equalities(isl.isl_union_set_copy(arg0.ptr))
        return union_set(ctx=ctx, ptr=res)
    def foreach_point(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = point(ctx=arg0.ctx, ptr=cb_arg0)
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_union_set_foreach_point(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        return res
    def foreach_set(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = set(ctx=arg0.ctx, ptr=cb_arg0)
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_union_set_foreach_set(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        return res
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_gist(isl.isl_union_set_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        return union_set(ctx=ctx, ptr=res)
    def gist_params(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_gist_params(isl.isl_union_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        return union_set(ctx=ctx, ptr=res)
    def identity(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_identity(isl.isl_union_set_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)
    def intersect(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_intersect(isl.isl_union_set_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        return union_set(ctx=ctx, ptr=res)
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_intersect_params(isl.isl_union_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        return union_set(ctx=ctx, ptr=res)
    def is_empty(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_is_empty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_strict_subset(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_is_strict_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_subset(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_is_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def lexmax(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_lexmax(isl.isl_union_set_copy(arg0.ptr))
        return union_set(ctx=ctx, ptr=res)
    def lexmin(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_lexmin(isl.isl_union_set_copy(arg0.ptr))
        return union_set(ctx=ctx, ptr=res)
    def polyhedral_hull(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_polyhedral_hull(isl.isl_union_set_copy(arg0.ptr))
        return union_set(ctx=ctx, ptr=res)
    def sample_point(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_sample_point(isl.isl_union_set_copy(arg0.ptr))
        return point(ctx=ctx, ptr=res)
    def subtract(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_subtract(isl.isl_union_set_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        return union_set(ctx=ctx, ptr=res)
    def union(arg0, arg1):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_set:
                arg1 = union_set(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_union(isl.isl_union_set_copy(arg0.ptr), isl.isl_union_set_copy(arg1.ptr))
        return union_set(ctx=ctx, ptr=res)
    def unwrap(arg0):
        try:
            if not arg0.__class__ is union_set:
                arg0 = union_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_set_unwrap(isl.isl_union_set_copy(arg0.ptr))
        return union_map(ctx=ctx, ptr=res)

isl.isl_union_set_from_basic_set.restype = c_void_p
isl.isl_union_set_from_basic_set.argtypes = [c_void_p]
isl.isl_union_set_from_set.restype = c_void_p
isl.isl_union_set_from_set.argtypes = [c_void_p]
isl.isl_union_set_from_point.restype = c_void_p
isl.isl_union_set_from_point.argtypes = [c_void_p]
isl.isl_union_set_read_from_str.restype = c_void_p
isl.isl_union_set_read_from_str.argtypes = [Context, c_char_p]
isl.isl_union_set_affine_hull.restype = c_void_p
isl.isl_union_set_affine_hull.argtypes = [c_void_p]
isl.isl_union_set_apply.restype = c_void_p
isl.isl_union_set_apply.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_coalesce.restype = c_void_p
isl.isl_union_set_coalesce.argtypes = [c_void_p]
isl.isl_union_set_compute_divs.restype = c_void_p
isl.isl_union_set_compute_divs.argtypes = [c_void_p]
isl.isl_union_set_detect_equalities.restype = c_void_p
isl.isl_union_set_detect_equalities.argtypes = [c_void_p]
isl.isl_union_set_foreach_point.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_union_set_foreach_set.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_union_set_gist.restype = c_void_p
isl.isl_union_set_gist.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_gist_params.restype = c_void_p
isl.isl_union_set_gist_params.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_identity.restype = c_void_p
isl.isl_union_set_identity.argtypes = [c_void_p]
isl.isl_union_set_intersect.restype = c_void_p
isl.isl_union_set_intersect.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_intersect_params.restype = c_void_p
isl.isl_union_set_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_is_empty.restype = c_bool
isl.isl_union_set_is_empty.argtypes = [c_void_p]
isl.isl_union_set_is_equal.restype = c_bool
isl.isl_union_set_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_is_strict_subset.restype = c_bool
isl.isl_union_set_is_strict_subset.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_is_subset.restype = c_bool
isl.isl_union_set_is_subset.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_lexmax.restype = c_void_p
isl.isl_union_set_lexmax.argtypes = [c_void_p]
isl.isl_union_set_lexmin.restype = c_void_p
isl.isl_union_set_lexmin.argtypes = [c_void_p]
isl.isl_union_set_polyhedral_hull.restype = c_void_p
isl.isl_union_set_polyhedral_hull.argtypes = [c_void_p]
isl.isl_union_set_sample_point.restype = c_void_p
isl.isl_union_set_sample_point.argtypes = [c_void_p]
isl.isl_union_set_subtract.restype = c_void_p
isl.isl_union_set_subtract.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_union.restype = c_void_p
isl.isl_union_set_union.argtypes = [c_void_p, c_void_p]
isl.isl_union_set_unwrap.restype = c_void_p
isl.isl_union_set_unwrap.argtypes = [c_void_p]
isl.isl_union_set_free.restype = c_void_p
isl.isl_union_set_free.argtypes = [c_void_p]
isl.isl_union_set_to_str.restype = POINTER(c_char)
isl.isl_union_set_to_str.argtypes = [c_void_p]

class set(union_set):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_set_read_from_str(self.ctx, args[0])
            return
        if len(args) == 1 and args[0].__class__ is basic_set:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_set_from_basic_set(isl.isl_basic_set_copy(args[0].ptr))
            return
        if len(args) == 1 and args[0].__class__ is point:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_set_from_point(isl.isl_point_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_set_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ptr = isl.isl_set_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.set("""%s""")' % s
        else:
            return 'isl.set("%s")' % s
    def affine_hull(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_affine_hull(isl.isl_set_copy(arg0.ptr))
        return basic_set(ctx=ctx, ptr=res)
    def apply(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is map:
                arg1 = map(arg1)
        except:
            return union_set(arg0).apply(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_apply(isl.isl_set_copy(arg0.ptr), isl.isl_map_copy(arg1.ptr))
        return set(ctx=ctx, ptr=res)
    def coalesce(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_coalesce(isl.isl_set_copy(arg0.ptr))
        return set(ctx=ctx, ptr=res)
    def complement(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_complement(isl.isl_set_copy(arg0.ptr))
        return set(ctx=ctx, ptr=res)
    def detect_equalities(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_detect_equalities(isl.isl_set_copy(arg0.ptr))
        return set(ctx=ctx, ptr=res)
    def flatten(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_flatten(isl.isl_set_copy(arg0.ptr))
        return set(ctx=ctx, ptr=res)
    def foreach_basic_set(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        exc_info = [None]
        fn = CFUNCTYPE(c_int, c_void_p, c_void_p)
        def cb_func(cb_arg0, cb_arg1):
            cb_arg0 = basic_set(ctx=arg0.ctx, ptr=cb_arg0)
            try:
                arg1(cb_arg0)
            except:
                import sys
                exc_info[0] = sys.exc_info()
                return -1
            return 0
        cb = fn(cb_func)
        ctx = arg0.ctx
        res = isl.isl_set_foreach_basic_set(arg0.ptr, cb, None)
        if exc_info[0] != None:
            raise (exc_info[0][0], exc_info[0][1], exc_info[0][2])
        return res
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_gist(isl.isl_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        return set(ctx=ctx, ptr=res)
    def identity(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_identity(isl.isl_set_copy(arg0.ptr))
        return map(ctx=ctx, ptr=res)
    def intersect(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).intersect(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_intersect(isl.isl_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        return set(ctx=ctx, ptr=res)
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).intersect_params(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_intersect_params(isl.isl_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        return set(ctx=ctx, ptr=res)
    def is_disjoint(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).is_disjoint(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_is_disjoint(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_empty(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_is_empty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).is_equal(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_strict_subset(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).is_strict_subset(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_is_strict_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_subset(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).is_subset(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_is_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_wrapping(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_is_wrapping(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def lexmax(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_lexmax(isl.isl_set_copy(arg0.ptr))
        return set(ctx=ctx, ptr=res)
    def lexmin(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_lexmin(isl.isl_set_copy(arg0.ptr))
        return set(ctx=ctx, ptr=res)
    def max_val(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return union_set(arg0).max_val(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_max_val(arg0.ptr, arg1.ptr)
        return val(ctx=ctx, ptr=res)
    def min_val(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is aff:
                arg1 = aff(arg1)
        except:
            return union_set(arg0).min_val(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_min_val(arg0.ptr, arg1.ptr)
        return val(ctx=ctx, ptr=res)
    def polyhedral_hull(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_polyhedral_hull(isl.isl_set_copy(arg0.ptr))
        return basic_set(ctx=ctx, ptr=res)
    def sample(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_sample(isl.isl_set_copy(arg0.ptr))
        return basic_set(ctx=ctx, ptr=res)
    def sample_point(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_sample_point(isl.isl_set_copy(arg0.ptr))
        return point(ctx=ctx, ptr=res)
    def subtract(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).subtract(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_subtract(isl.isl_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        return set(ctx=ctx, ptr=res)
    def union(arg0, arg1):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is set:
                arg1 = set(arg1)
        except:
            return union_set(arg0).union(arg1)
        ctx = arg0.ctx
        res = isl.isl_set_union(isl.isl_set_copy(arg0.ptr), isl.isl_set_copy(arg1.ptr))
        return set(ctx=ctx, ptr=res)
    def unshifted_simple_hull(arg0):
        try:
            if not arg0.__class__ is set:
                arg0 = set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_set_unshifted_simple_hull(isl.isl_set_copy(arg0.ptr))
        return basic_set(ctx=ctx, ptr=res)

isl.isl_set_read_from_str.restype = c_void_p
isl.isl_set_read_from_str.argtypes = [Context, c_char_p]
isl.isl_set_from_basic_set.restype = c_void_p
isl.isl_set_from_basic_set.argtypes = [c_void_p]
isl.isl_set_from_point.restype = c_void_p
isl.isl_set_from_point.argtypes = [c_void_p]
isl.isl_set_affine_hull.restype = c_void_p
isl.isl_set_affine_hull.argtypes = [c_void_p]
isl.isl_set_apply.restype = c_void_p
isl.isl_set_apply.argtypes = [c_void_p, c_void_p]
isl.isl_set_coalesce.restype = c_void_p
isl.isl_set_coalesce.argtypes = [c_void_p]
isl.isl_set_complement.restype = c_void_p
isl.isl_set_complement.argtypes = [c_void_p]
isl.isl_set_detect_equalities.restype = c_void_p
isl.isl_set_detect_equalities.argtypes = [c_void_p]
isl.isl_set_flatten.restype = c_void_p
isl.isl_set_flatten.argtypes = [c_void_p]
isl.isl_set_foreach_basic_set.argtypes = [c_void_p, c_void_p, c_void_p]
isl.isl_set_gist.restype = c_void_p
isl.isl_set_gist.argtypes = [c_void_p, c_void_p]
isl.isl_set_identity.restype = c_void_p
isl.isl_set_identity.argtypes = [c_void_p]
isl.isl_set_intersect.restype = c_void_p
isl.isl_set_intersect.argtypes = [c_void_p, c_void_p]
isl.isl_set_intersect_params.restype = c_void_p
isl.isl_set_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_set_is_disjoint.restype = c_bool
isl.isl_set_is_disjoint.argtypes = [c_void_p, c_void_p]
isl.isl_set_is_empty.restype = c_bool
isl.isl_set_is_empty.argtypes = [c_void_p]
isl.isl_set_is_equal.restype = c_bool
isl.isl_set_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_set_is_strict_subset.restype = c_bool
isl.isl_set_is_strict_subset.argtypes = [c_void_p, c_void_p]
isl.isl_set_is_subset.restype = c_bool
isl.isl_set_is_subset.argtypes = [c_void_p, c_void_p]
isl.isl_set_is_wrapping.restype = c_bool
isl.isl_set_is_wrapping.argtypes = [c_void_p]
isl.isl_set_lexmax.restype = c_void_p
isl.isl_set_lexmax.argtypes = [c_void_p]
isl.isl_set_lexmin.restype = c_void_p
isl.isl_set_lexmin.argtypes = [c_void_p]
isl.isl_set_max_val.restype = c_void_p
isl.isl_set_max_val.argtypes = [c_void_p, c_void_p]
isl.isl_set_min_val.restype = c_void_p
isl.isl_set_min_val.argtypes = [c_void_p, c_void_p]
isl.isl_set_polyhedral_hull.restype = c_void_p
isl.isl_set_polyhedral_hull.argtypes = [c_void_p]
isl.isl_set_sample.restype = c_void_p
isl.isl_set_sample.argtypes = [c_void_p]
isl.isl_set_sample_point.restype = c_void_p
isl.isl_set_sample_point.argtypes = [c_void_p]
isl.isl_set_subtract.restype = c_void_p
isl.isl_set_subtract.argtypes = [c_void_p, c_void_p]
isl.isl_set_union.restype = c_void_p
isl.isl_set_union.argtypes = [c_void_p, c_void_p]
isl.isl_set_unshifted_simple_hull.restype = c_void_p
isl.isl_set_unshifted_simple_hull.argtypes = [c_void_p]
isl.isl_set_free.restype = c_void_p
isl.isl_set_free.argtypes = [c_void_p]
isl.isl_set_to_str.restype = POINTER(c_char)
isl.isl_set_to_str.argtypes = [c_void_p]

class basic_set(set):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_basic_set_read_from_str(self.ctx, args[0])
            return
        if len(args) == 1 and args[0].__class__ is point:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_basic_set_from_point(isl.isl_point_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_basic_set_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ptr = isl.isl_basic_set_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.basic_set("""%s""")' % s
        else:
            return 'isl.basic_set("%s")' % s
    def affine_hull(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_affine_hull(isl.isl_basic_set_copy(arg0.ptr))
        return basic_set(ctx=ctx, ptr=res)
    def apply(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_map:
                arg1 = basic_map(arg1)
        except:
            return set(arg0).apply(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_apply(isl.isl_basic_set_copy(arg0.ptr), isl.isl_basic_map_copy(arg1.ptr))
        return basic_set(ctx=ctx, ptr=res)
    def detect_equalities(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_detect_equalities(isl.isl_basic_set_copy(arg0.ptr))
        return basic_set(ctx=ctx, ptr=res)
    def flatten(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_flatten(isl.isl_basic_set_copy(arg0.ptr))
        return basic_set(ctx=ctx, ptr=res)
    def gist(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return set(arg0).gist(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_gist(isl.isl_basic_set_copy(arg0.ptr), isl.isl_basic_set_copy(arg1.ptr))
        return basic_set(ctx=ctx, ptr=res)
    def intersect(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return set(arg0).intersect(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_intersect(isl.isl_basic_set_copy(arg0.ptr), isl.isl_basic_set_copy(arg1.ptr))
        return basic_set(ctx=ctx, ptr=res)
    def intersect_params(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return set(arg0).intersect_params(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_intersect_params(isl.isl_basic_set_copy(arg0.ptr), isl.isl_basic_set_copy(arg1.ptr))
        return basic_set(ctx=ctx, ptr=res)
    def is_empty(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_is_empty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_equal(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return set(arg0).is_equal(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_is_equal(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_subset(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return set(arg0).is_subset(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_is_subset(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_wrapping(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_is_wrapping(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def lexmax(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_lexmax(isl.isl_basic_set_copy(arg0.ptr))
        return set(ctx=ctx, ptr=res)
    def lexmin(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_lexmin(isl.isl_basic_set_copy(arg0.ptr))
        return set(ctx=ctx, ptr=res)
    def sample(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_sample(isl.isl_basic_set_copy(arg0.ptr))
        return basic_set(ctx=ctx, ptr=res)
    def sample_point(arg0):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_basic_set_sample_point(isl.isl_basic_set_copy(arg0.ptr))
        return point(ctx=ctx, ptr=res)
    def union(arg0, arg1):
        try:
            if not arg0.__class__ is basic_set:
                arg0 = basic_set(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is basic_set:
                arg1 = basic_set(arg1)
        except:
            return set(arg0).union(arg1)
        ctx = arg0.ctx
        res = isl.isl_basic_set_union(isl.isl_basic_set_copy(arg0.ptr), isl.isl_basic_set_copy(arg1.ptr))
        return set(ctx=ctx, ptr=res)

isl.isl_basic_set_read_from_str.restype = c_void_p
isl.isl_basic_set_read_from_str.argtypes = [Context, c_char_p]
isl.isl_basic_set_from_point.restype = c_void_p
isl.isl_basic_set_from_point.argtypes = [c_void_p]
isl.isl_basic_set_affine_hull.restype = c_void_p
isl.isl_basic_set_affine_hull.argtypes = [c_void_p]
isl.isl_basic_set_apply.restype = c_void_p
isl.isl_basic_set_apply.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_detect_equalities.restype = c_void_p
isl.isl_basic_set_detect_equalities.argtypes = [c_void_p]
isl.isl_basic_set_flatten.restype = c_void_p
isl.isl_basic_set_flatten.argtypes = [c_void_p]
isl.isl_basic_set_gist.restype = c_void_p
isl.isl_basic_set_gist.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_intersect.restype = c_void_p
isl.isl_basic_set_intersect.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_intersect_params.restype = c_void_p
isl.isl_basic_set_intersect_params.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_is_empty.restype = c_bool
isl.isl_basic_set_is_empty.argtypes = [c_void_p]
isl.isl_basic_set_is_equal.restype = c_bool
isl.isl_basic_set_is_equal.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_is_subset.restype = c_bool
isl.isl_basic_set_is_subset.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_is_wrapping.restype = c_bool
isl.isl_basic_set_is_wrapping.argtypes = [c_void_p]
isl.isl_basic_set_lexmax.restype = c_void_p
isl.isl_basic_set_lexmax.argtypes = [c_void_p]
isl.isl_basic_set_lexmin.restype = c_void_p
isl.isl_basic_set_lexmin.argtypes = [c_void_p]
isl.isl_basic_set_sample.restype = c_void_p
isl.isl_basic_set_sample.argtypes = [c_void_p]
isl.isl_basic_set_sample_point.restype = c_void_p
isl.isl_basic_set_sample_point.argtypes = [c_void_p]
isl.isl_basic_set_union.restype = c_void_p
isl.isl_basic_set_union.argtypes = [c_void_p, c_void_p]
isl.isl_basic_set_free.restype = c_void_p
isl.isl_basic_set_free.argtypes = [c_void_p]
isl.isl_basic_set_to_str.restype = POINTER(c_char)
isl.isl_basic_set_to_str.argtypes = [c_void_p]

class multi_val(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_multi_val_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        ptr = isl.isl_multi_val_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.multi_val("""%s""")' % s
        else:
            return 'isl.multi_val("%s")' % s
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_add(isl.isl_multi_val_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        return multi_val(ctx=ctx, ptr=res)
    def flat_range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_flat_range_product(isl.isl_multi_val_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        return multi_val(ctx=ctx, ptr=res)
    def product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_product(isl.isl_multi_val_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        return multi_val(ctx=ctx, ptr=res)
    def range_product(arg0, arg1):
        try:
            if not arg0.__class__ is multi_val:
                arg0 = multi_val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is multi_val:
                arg1 = multi_val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_multi_val_range_product(isl.isl_multi_val_copy(arg0.ptr), isl.isl_multi_val_copy(arg1.ptr))
        return multi_val(ctx=ctx, ptr=res)

isl.isl_multi_val_add.restype = c_void_p
isl.isl_multi_val_add.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_flat_range_product.restype = c_void_p
isl.isl_multi_val_flat_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_product.restype = c_void_p
isl.isl_multi_val_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_range_product.restype = c_void_p
isl.isl_multi_val_range_product.argtypes = [c_void_p, c_void_p]
isl.isl_multi_val_free.restype = c_void_p
isl.isl_multi_val_free.argtypes = [c_void_p]
isl.isl_multi_val_to_str.restype = POINTER(c_char)
isl.isl_multi_val_to_str.argtypes = [c_void_p]

class point(basic_set):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_point_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is point:
                arg0 = point(arg0)
        except:
            raise
        ptr = isl.isl_point_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.point("""%s""")' % s
        else:
            return 'isl.point("%s")' % s

isl.isl_point_free.argtypes = [c_void_p]
isl.isl_point_to_str.restype = POINTER(c_char)
isl.isl_point_to_str.argtypes = [c_void_p]

class schedule(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_schedule_read_from_str(self.ctx, args[0])
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule:
                arg0 = schedule(arg0)
        except:
            raise
        ptr = isl.isl_schedule_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule("""%s""")' % s
        else:
            return 'isl.schedule("%s")' % s
    def get_map(arg0):
        try:
            if not arg0.__class__ is schedule:
                arg0 = schedule(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_get_map(arg0.ptr)
        return union_map(ctx=ctx, ptr=res)
    def get_root(arg0):
        try:
            if not arg0.__class__ is schedule:
                arg0 = schedule(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_get_root(arg0.ptr)
        return schedule_node(ctx=ctx, ptr=res)
    def pullback(arg0, arg1):
        if arg1.__class__ is union_pw_multi_aff:
            res = isl.isl_schedule_pullback_union_pw_multi_aff(isl.isl_schedule_copy(arg0.ptr), isl.isl_union_pw_multi_aff_copy(arg1.ptr))
            return schedule(ctx=arg0.ctx, ptr=res)

isl.isl_schedule_read_from_str.restype = c_void_p
isl.isl_schedule_read_from_str.argtypes = [Context, c_char_p]
isl.isl_schedule_get_map.restype = c_void_p
isl.isl_schedule_get_map.argtypes = [c_void_p]
isl.isl_schedule_get_root.restype = c_void_p
isl.isl_schedule_get_root.argtypes = [c_void_p]
isl.isl_schedule_pullback_union_pw_multi_aff.restype = c_void_p
isl.isl_schedule_pullback_union_pw_multi_aff.argtypes = [c_void_p, c_void_p]
isl.isl_schedule_free.restype = c_void_p
isl.isl_schedule_free.argtypes = [c_void_p]
isl.isl_schedule_to_str.restype = POINTER(c_char)
isl.isl_schedule_to_str.argtypes = [c_void_p]

class schedule_constraints(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_schedule_constraints_read_from_str(self.ctx, args[0])
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_constraints_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ptr = isl.isl_schedule_constraints_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_constraints("""%s""")' % s
        else:
            return 'isl.schedule_constraints("%s")' % s
    def get_coincidence(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_coincidence(arg0.ptr)
        return union_map(ctx=ctx, ptr=res)
    def get_conditional_validity(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_conditional_validity(arg0.ptr)
        return union_map(ctx=ctx, ptr=res)
    def get_conditional_validity_condition(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_conditional_validity_condition(arg0.ptr)
        return union_map(ctx=ctx, ptr=res)
    def get_context(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_context(arg0.ptr)
        return set(ctx=ctx, ptr=res)
    def get_domain(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_domain(arg0.ptr)
        return union_set(ctx=ctx, ptr=res)
    def get_proximity(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_proximity(arg0.ptr)
        return union_map(ctx=ctx, ptr=res)
    def get_validity(arg0):
        try:
            if not arg0.__class__ is schedule_constraints:
                arg0 = schedule_constraints(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_constraints_get_validity(arg0.ptr)
        return union_map(ctx=ctx, ptr=res)

isl.isl_schedule_constraints_read_from_str.restype = c_void_p
isl.isl_schedule_constraints_read_from_str.argtypes = [Context, c_char_p]
isl.isl_schedule_constraints_get_coincidence.restype = c_void_p
isl.isl_schedule_constraints_get_coincidence.argtypes = [c_void_p]
isl.isl_schedule_constraints_get_conditional_validity.restype = c_void_p
isl.isl_schedule_constraints_get_conditional_validity.argtypes = [c_void_p]
isl.isl_schedule_constraints_get_conditional_validity_condition.restype = c_void_p
isl.isl_schedule_constraints_get_conditional_validity_condition.argtypes = [c_void_p]
isl.isl_schedule_constraints_get_context.restype = c_void_p
isl.isl_schedule_constraints_get_context.argtypes = [c_void_p]
isl.isl_schedule_constraints_get_domain.restype = c_void_p
isl.isl_schedule_constraints_get_domain.argtypes = [c_void_p]
isl.isl_schedule_constraints_get_proximity.restype = c_void_p
isl.isl_schedule_constraints_get_proximity.argtypes = [c_void_p]
isl.isl_schedule_constraints_get_validity.restype = c_void_p
isl.isl_schedule_constraints_get_validity.argtypes = [c_void_p]
isl.isl_schedule_constraints_free.restype = c_void_p
isl.isl_schedule_constraints_free.argtypes = [c_void_p]
isl.isl_schedule_constraints_to_str.restype = POINTER(c_char)
isl.isl_schedule_constraints_to_str.argtypes = [c_void_p]

class schedule_node(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_schedule_node_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ptr = isl.isl_schedule_node_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.schedule_node("""%s""")' % s
        else:
            return 'isl.schedule_node("%s")' % s
    def band_member_get_coincident(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_member_get_coincident(arg0.ptr, arg1)
        if res < 0:
            raise
        return bool(res)
    def band_member_set_coincident(arg0, arg1, arg2):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_band_member_set_coincident(isl.isl_schedule_node_copy(arg0.ptr), arg1, arg2)
        return schedule_node(ctx=ctx, ptr=res)
    def child(arg0, arg1):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_child(isl.isl_schedule_node_copy(arg0.ptr), arg1)
        return schedule_node(ctx=ctx, ptr=res)
    def get_prefix_schedule_multi_union_pw_aff(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(arg0.ptr)
        return multi_union_pw_aff(ctx=ctx, ptr=res)
    def get_prefix_schedule_union_map(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_get_prefix_schedule_union_map(arg0.ptr)
        return union_map(ctx=ctx, ptr=res)
    def get_prefix_schedule_union_pw_multi_aff(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(arg0.ptr)
        return union_pw_multi_aff(ctx=ctx, ptr=res)
    def get_schedule(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_get_schedule(arg0.ptr)
        return schedule(ctx=ctx, ptr=res)
    def parent(arg0):
        try:
            if not arg0.__class__ is schedule_node:
                arg0 = schedule_node(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_schedule_node_parent(isl.isl_schedule_node_copy(arg0.ptr))
        return schedule_node(ctx=ctx, ptr=res)

isl.isl_schedule_node_band_member_get_coincident.restype = c_bool
isl.isl_schedule_node_band_member_get_coincident.argtypes = [c_void_p, c_int]
isl.isl_schedule_node_band_member_set_coincident.restype = c_void_p
isl.isl_schedule_node_band_member_set_coincident.argtypes = [c_void_p, c_int, c_int]
isl.isl_schedule_node_child.restype = c_void_p
isl.isl_schedule_node_child.argtypes = [c_void_p, c_int]
isl.isl_schedule_node_get_prefix_schedule_multi_union_pw_aff.restype = c_void_p
isl.isl_schedule_node_get_prefix_schedule_multi_union_pw_aff.argtypes = [c_void_p]
isl.isl_schedule_node_get_prefix_schedule_union_map.restype = c_void_p
isl.isl_schedule_node_get_prefix_schedule_union_map.argtypes = [c_void_p]
isl.isl_schedule_node_get_prefix_schedule_union_pw_multi_aff.restype = c_void_p
isl.isl_schedule_node_get_prefix_schedule_union_pw_multi_aff.argtypes = [c_void_p]
isl.isl_schedule_node_get_schedule.restype = c_void_p
isl.isl_schedule_node_get_schedule.argtypes = [c_void_p]
isl.isl_schedule_node_parent.restype = c_void_p
isl.isl_schedule_node_parent.argtypes = [c_void_p]
isl.isl_schedule_node_free.restype = c_void_p
isl.isl_schedule_node_free.argtypes = [c_void_p]
isl.isl_schedule_node_to_str.restype = POINTER(c_char)
isl.isl_schedule_node_to_str.argtypes = [c_void_p]

class union_access_info(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and args[0].__class__ is union_map:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_union_access_info_from_sink(isl.isl_union_map_copy(args[0].ptr))
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_access_info_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_access_info:
                arg0 = union_access_info(arg0)
        except:
            raise
        ptr = isl.isl_union_access_info_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_access_info("""%s""")' % s
        else:
            return 'isl.union_access_info("%s")' % s
    def compute_flow(arg0):
        try:
            if not arg0.__class__ is union_access_info:
                arg0 = union_access_info(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_access_info_compute_flow(isl.isl_union_access_info_copy(arg0.ptr))
        return union_flow(ctx=ctx, ptr=res)
    def set_may_source(arg0, arg1):
        try:
            if not arg0.__class__ is union_access_info:
                arg0 = union_access_info(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_access_info_set_may_source(isl.isl_union_access_info_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        return union_access_info(ctx=ctx, ptr=res)
    def set_must_source(arg0, arg1):
        try:
            if not arg0.__class__ is union_access_info:
                arg0 = union_access_info(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_access_info_set_must_source(isl.isl_union_access_info_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        return union_access_info(ctx=ctx, ptr=res)
    def set_schedule(arg0, arg1):
        try:
            if not arg0.__class__ is union_access_info:
                arg0 = union_access_info(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is schedule:
                arg1 = schedule(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_access_info_set_schedule(isl.isl_union_access_info_copy(arg0.ptr), isl.isl_schedule_copy(arg1.ptr))
        return union_access_info(ctx=ctx, ptr=res)
    def set_schedule_map(arg0, arg1):
        try:
            if not arg0.__class__ is union_access_info:
                arg0 = union_access_info(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is union_map:
                arg1 = union_map(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_access_info_set_schedule_map(isl.isl_union_access_info_copy(arg0.ptr), isl.isl_union_map_copy(arg1.ptr))
        return union_access_info(ctx=ctx, ptr=res)

isl.isl_union_access_info_from_sink.restype = c_void_p
isl.isl_union_access_info_from_sink.argtypes = [c_void_p]
isl.isl_union_access_info_compute_flow.restype = c_void_p
isl.isl_union_access_info_compute_flow.argtypes = [c_void_p]
isl.isl_union_access_info_set_may_source.restype = c_void_p
isl.isl_union_access_info_set_may_source.argtypes = [c_void_p, c_void_p]
isl.isl_union_access_info_set_must_source.restype = c_void_p
isl.isl_union_access_info_set_must_source.argtypes = [c_void_p, c_void_p]
isl.isl_union_access_info_set_schedule.restype = c_void_p
isl.isl_union_access_info_set_schedule.argtypes = [c_void_p, c_void_p]
isl.isl_union_access_info_set_schedule_map.restype = c_void_p
isl.isl_union_access_info_set_schedule_map.argtypes = [c_void_p, c_void_p]
isl.isl_union_access_info_free.restype = c_void_p
isl.isl_union_access_info_free.argtypes = [c_void_p]
isl.isl_union_access_info_to_str.restype = POINTER(c_char)
isl.isl_union_access_info_to_str.argtypes = [c_void_p]

class union_flow(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_union_flow_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ptr = isl.isl_union_flow_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.union_flow("""%s""")' % s
        else:
            return 'isl.union_flow("%s")' % s
    def get_full_may_dependence(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_flow_get_full_may_dependence(arg0.ptr)
        return union_map(ctx=ctx, ptr=res)
    def get_full_must_dependence(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_flow_get_full_must_dependence(arg0.ptr)
        return union_map(ctx=ctx, ptr=res)
    def get_may_dependence(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_flow_get_may_dependence(arg0.ptr)
        return union_map(ctx=ctx, ptr=res)
    def get_may_no_source(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_flow_get_may_no_source(arg0.ptr)
        return union_map(ctx=ctx, ptr=res)
    def get_must_dependence(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_flow_get_must_dependence(arg0.ptr)
        return union_map(ctx=ctx, ptr=res)
    def get_must_no_source(arg0):
        try:
            if not arg0.__class__ is union_flow:
                arg0 = union_flow(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_union_flow_get_must_no_source(arg0.ptr)
        return union_map(ctx=ctx, ptr=res)

isl.isl_union_flow_get_full_may_dependence.restype = c_void_p
isl.isl_union_flow_get_full_may_dependence.argtypes = [c_void_p]
isl.isl_union_flow_get_full_must_dependence.restype = c_void_p
isl.isl_union_flow_get_full_must_dependence.argtypes = [c_void_p]
isl.isl_union_flow_get_may_dependence.restype = c_void_p
isl.isl_union_flow_get_may_dependence.argtypes = [c_void_p]
isl.isl_union_flow_get_may_no_source.restype = c_void_p
isl.isl_union_flow_get_may_no_source.argtypes = [c_void_p]
isl.isl_union_flow_get_must_dependence.restype = c_void_p
isl.isl_union_flow_get_must_dependence.argtypes = [c_void_p]
isl.isl_union_flow_get_must_no_source.restype = c_void_p
isl.isl_union_flow_get_must_no_source.argtypes = [c_void_p]
isl.isl_union_flow_free.restype = c_void_p
isl.isl_union_flow_free.argtypes = [c_void_p]
isl.isl_union_flow_to_str.restype = POINTER(c_char)
isl.isl_union_flow_to_str.argtypes = [c_void_p]

class val(object):
    def __init__(self, *args, **keywords):
        if "ptr" in keywords:
            self.ctx = keywords["ctx"]
            self.ptr = keywords["ptr"]
            return
        if len(args) == 1 and type(args[0]) == int:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_val_int_from_si(self.ctx, args[0])
            return
        if len(args) == 1 and type(args[0]) == str:
            self.ctx = Context.getDefaultInstance()
            self.ptr = isl.isl_val_read_from_str(self.ctx, args[0])
            return
        raise Error
    def __del__(self):
        if hasattr(self, 'ptr'):
            isl.isl_val_free(self.ptr)
    def __str__(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ptr = isl.isl_val_to_str(arg0.ptr)
        res = str(cast(ptr, c_char_p).value)
        libc.free(ptr)
        return res
    def __repr__(self):
        s = str(self)
        if '"' in s:
            return 'isl.val("""%s""")' % s
        else:
            return 'isl.val("%s")' % s
    def abs(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_abs(isl.isl_val_copy(arg0.ptr))
        return val(ctx=ctx, ptr=res)
    def abs_eq(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_abs_eq(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def add(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_add(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        return val(ctx=ctx, ptr=res)
    def ceil(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_ceil(isl.isl_val_copy(arg0.ptr))
        return val(ctx=ctx, ptr=res)
    def cmp_si(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_cmp_si(arg0.ptr, arg1)
        return res
    def div(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_div(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        return val(ctx=ctx, ptr=res)
    def eq(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_eq(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def floor(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_floor(isl.isl_val_copy(arg0.ptr))
        return val(ctx=ctx, ptr=res)
    def gcd(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_gcd(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        return val(ctx=ctx, ptr=res)
    def ge(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_ge(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def gt(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_gt(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    @staticmethod
    def infty():
        ctx = Context.getDefaultInstance()
        res = isl.isl_val_infty(ctx)
        return val(ctx=ctx, ptr=res)
    def inv(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_inv(isl.isl_val_copy(arg0.ptr))
        return val(ctx=ctx, ptr=res)
    def is_divisible_by(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_divisible_by(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_infty(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_infty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_int(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_int(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_nan(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_nan(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_neg(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_neg(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_neginfty(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_neginfty(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_negone(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_negone(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_nonneg(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_nonneg(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_nonpos(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_nonpos(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_one(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_one(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_pos(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_pos(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_rat(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_rat(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def is_zero(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_is_zero(arg0.ptr)
        if res < 0:
            raise
        return bool(res)
    def le(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_le(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def lt(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_lt(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def max(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_max(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        return val(ctx=ctx, ptr=res)
    def min(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_min(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        return val(ctx=ctx, ptr=res)
    def mod(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_mod(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        return val(ctx=ctx, ptr=res)
    def mul(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_mul(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        return val(ctx=ctx, ptr=res)
    @staticmethod
    def nan():
        ctx = Context.getDefaultInstance()
        res = isl.isl_val_nan(ctx)
        return val(ctx=ctx, ptr=res)
    def ne(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_ne(arg0.ptr, arg1.ptr)
        if res < 0:
            raise
        return bool(res)
    def neg(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_neg(isl.isl_val_copy(arg0.ptr))
        return val(ctx=ctx, ptr=res)
    @staticmethod
    def neginfty():
        ctx = Context.getDefaultInstance()
        res = isl.isl_val_neginfty(ctx)
        return val(ctx=ctx, ptr=res)
    @staticmethod
    def negone():
        ctx = Context.getDefaultInstance()
        res = isl.isl_val_negone(ctx)
        return val(ctx=ctx, ptr=res)
    @staticmethod
    def one():
        ctx = Context.getDefaultInstance()
        res = isl.isl_val_one(ctx)
        return val(ctx=ctx, ptr=res)
    def sgn(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_sgn(arg0.ptr)
        return res
    def sub(arg0, arg1):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        try:
            if not arg1.__class__ is val:
                arg1 = val(arg1)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_sub(isl.isl_val_copy(arg0.ptr), isl.isl_val_copy(arg1.ptr))
        return val(ctx=ctx, ptr=res)
    def trunc(arg0):
        try:
            if not arg0.__class__ is val:
                arg0 = val(arg0)
        except:
            raise
        ctx = arg0.ctx
        res = isl.isl_val_trunc(isl.isl_val_copy(arg0.ptr))
        return val(ctx=ctx, ptr=res)
    @staticmethod
    def zero():
        ctx = Context.getDefaultInstance()
        res = isl.isl_val_zero(ctx)
        return val(ctx=ctx, ptr=res)

isl.isl_val_int_from_si.restype = c_void_p
isl.isl_val_int_from_si.argtypes = [Context, c_int]
isl.isl_val_read_from_str.restype = c_void_p
isl.isl_val_read_from_str.argtypes = [Context, c_char_p]
isl.isl_val_abs.restype = c_void_p
isl.isl_val_abs.argtypes = [c_void_p]
isl.isl_val_abs_eq.restype = c_bool
isl.isl_val_abs_eq.argtypes = [c_void_p, c_void_p]
isl.isl_val_add.restype = c_void_p
isl.isl_val_add.argtypes = [c_void_p, c_void_p]
isl.isl_val_ceil.restype = c_void_p
isl.isl_val_ceil.argtypes = [c_void_p]
isl.isl_val_cmp_si.argtypes = [c_void_p, c_int]
isl.isl_val_div.restype = c_void_p
isl.isl_val_div.argtypes = [c_void_p, c_void_p]
isl.isl_val_eq.restype = c_bool
isl.isl_val_eq.argtypes = [c_void_p, c_void_p]
isl.isl_val_floor.restype = c_void_p
isl.isl_val_floor.argtypes = [c_void_p]
isl.isl_val_gcd.restype = c_void_p
isl.isl_val_gcd.argtypes = [c_void_p, c_void_p]
isl.isl_val_ge.restype = c_bool
isl.isl_val_ge.argtypes = [c_void_p, c_void_p]
isl.isl_val_gt.restype = c_bool
isl.isl_val_gt.argtypes = [c_void_p, c_void_p]
isl.isl_val_infty.restype = c_void_p
isl.isl_val_infty.argtypes = [Context]
isl.isl_val_inv.restype = c_void_p
isl.isl_val_inv.argtypes = [c_void_p]
isl.isl_val_is_divisible_by.restype = c_bool
isl.isl_val_is_divisible_by.argtypes = [c_void_p, c_void_p]
isl.isl_val_is_infty.restype = c_bool
isl.isl_val_is_infty.argtypes = [c_void_p]
isl.isl_val_is_int.restype = c_bool
isl.isl_val_is_int.argtypes = [c_void_p]
isl.isl_val_is_nan.restype = c_bool
isl.isl_val_is_nan.argtypes = [c_void_p]
isl.isl_val_is_neg.restype = c_bool
isl.isl_val_is_neg.argtypes = [c_void_p]
isl.isl_val_is_neginfty.restype = c_bool
isl.isl_val_is_neginfty.argtypes = [c_void_p]
isl.isl_val_is_negone.restype = c_bool
isl.isl_val_is_negone.argtypes = [c_void_p]
isl.isl_val_is_nonneg.restype = c_bool
isl.isl_val_is_nonneg.argtypes = [c_void_p]
isl.isl_val_is_nonpos.restype = c_bool
isl.isl_val_is_nonpos.argtypes = [c_void_p]
isl.isl_val_is_one.restype = c_bool
isl.isl_val_is_one.argtypes = [c_void_p]
isl.isl_val_is_pos.restype = c_bool
isl.isl_val_is_pos.argtypes = [c_void_p]
isl.isl_val_is_rat.restype = c_bool
isl.isl_val_is_rat.argtypes = [c_void_p]
isl.isl_val_is_zero.restype = c_bool
isl.isl_val_is_zero.argtypes = [c_void_p]
isl.isl_val_le.restype = c_bool
isl.isl_val_le.argtypes = [c_void_p, c_void_p]
isl.isl_val_lt.restype = c_bool
isl.isl_val_lt.argtypes = [c_void_p, c_void_p]
isl.isl_val_max.restype = c_void_p
isl.isl_val_max.argtypes = [c_void_p, c_void_p]
isl.isl_val_min.restype = c_void_p
isl.isl_val_min.argtypes = [c_void_p, c_void_p]
isl.isl_val_mod.restype = c_void_p
isl.isl_val_mod.argtypes = [c_void_p, c_void_p]
isl.isl_val_mul.restype = c_void_p
isl.isl_val_mul.argtypes = [c_void_p, c_void_p]
isl.isl_val_nan.restype = c_void_p
isl.isl_val_nan.argtypes = [Context]
isl.isl_val_ne.restype = c_bool
isl.isl_val_ne.argtypes = [c_void_p, c_void_p]
isl.isl_val_neg.restype = c_void_p
isl.isl_val_neg.argtypes = [c_void_p]
isl.isl_val_neginfty.restype = c_void_p
isl.isl_val_neginfty.argtypes = [Context]
isl.isl_val_negone.restype = c_void_p
isl.isl_val_negone.argtypes = [Context]
isl.isl_val_one.restype = c_void_p
isl.isl_val_one.argtypes = [Context]
isl.isl_val_sgn.argtypes = [c_void_p]
isl.isl_val_sub.restype = c_void_p
isl.isl_val_sub.argtypes = [c_void_p, c_void_p]
isl.isl_val_trunc.restype = c_void_p
isl.isl_val_trunc.argtypes = [c_void_p]
isl.isl_val_zero.restype = c_void_p
isl.isl_val_zero.argtypes = [Context]
isl.isl_val_free.restype = c_void_p
isl.isl_val_free.argtypes = [c_void_p]
isl.isl_val_to_str.restype = POINTER(c_char)
isl.isl_val_to_str.argtypes = [c_void_p]
