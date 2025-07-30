import gdb
import re

# GDB Pretty Printers for most isl objects
class IslObjectPrinter:
	"""Print an isl object"""
	def __init__ (self, val, type):
                self.val = val
                self.type = type

	def to_string (self):
                # Cast val to a void pointer to stop gdb using this pretty
                # printer for the pointer which would lead to an infinite loop.
                void_ptr = gdb.lookup_type('void').pointer()
                value = str(self.val.cast(void_ptr))
                printer = gdb.parse_and_eval("isl_printer_to_str(isl_"
                                             + str(self.type)
					     + "_get_ctx(" + value + "))")
                printer = gdb.parse_and_eval("isl_printer_print_"
                                             + str(self.type) + "("
                                             + str(printer) + ", "
                                             + value + ")")
                string = gdb.parse_and_eval("(char*)isl_printer_get_str("
                                            + str(printer) + ")")
                gdb.parse_and_eval("isl_printer_free(" + str(printer) + ")")
                return string

	def display_hint (self):
                return 'string'

class IslIntPrinter:
	"""Print an isl_int """
	def __init__ (self, val):
                self.val = val

	def to_string (self):
                # Cast val to a void pointer to stop gdb using this pretty
                # printer for the pointer which would lead to an infinite loop.
                void_ptr = gdb.lookup_type('void').pointer()
                value = str(self.val.cast(void_ptr))

                context = gdb.parse_and_eval("isl_ctx_alloc()")
                printer = gdb.parse_and_eval("isl_printer_to_str("
                                             + str(context) + ")")
                printer = gdb.parse_and_eval("isl_printer_print_isl_int("
                                             + str(printer) + ", "
                                             + value + ")")
                string = gdb.parse_and_eval("(char*)isl_printer_get_str("
                                            + str(printer) + ")")
                gdb.parse_and_eval("isl_printer_free(" + str(printer) + ")")
                gdb.parse_and_eval("isl_ctx_free(" + str(context) + ")")
                return string

	def display_hint (self):
                return 'string'

class IslPrintCommand (gdb.Command):
        """Print an isl value."""
        def __init__ (self):
                super (IslPrintCommand, self).__init__ ("islprint",
                                                        gdb.COMMAND_OBSCURE)
        def invoke (self, arg, from_tty):
                arg = gdb.parse_and_eval(arg);
                printer = str_lookup_function(arg)

                if printer == None:
                        print "No isl printer for this type"
                        return

                print printer.to_string()

IslPrintCommand()

def str_lookup_function (val):
	if val.type.code != gdb.TYPE_CODE_PTR:
		if str(val.type) == "isl_int":
			return IslIntPrinter(val)
                else:
                        return None

	lookup_tag = val.type.target()
	regex = re.compile ("^isl_(.*)$")

	if lookup_tag == None:
		return None

	m = regex.match (str(lookup_tag))

	if m:
                # Those types of printers defined in isl.
                if m.group(1) in ["basic_set", "set", "union_set", "basic_map",
                                  "map", "union_map", "qpolynomial",
                                  "pw_qpolynomial", "pw_qpolynomial_fold",
                                  "union_pw_qpolynomial",
                                  "union_pw_qpolynomial_fold"]:
                        return IslObjectPrinter(val, m.group(1))
        return None

# Do not register the pretty printer.
# gdb.current_objfile().pretty_printers.append(str_lookup_function)
