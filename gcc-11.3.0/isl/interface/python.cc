/*
 * Copyright 2011,2015 Sven Verdoolaege. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 * 
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY SVEN VERDOOLAEGE ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SVEN VERDOOLAEGE OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * The views and conclusions contained in the software and documentation
 * are those of the authors and should not be interpreted as
 * representing official policies, either expressed or implied, of
 * Sven Verdoolaege.
 */ 

#include "isl_config.h"

#include <stdio.h>
#include <iostream>
#include <map>
#include <vector>
#include <clang/AST/Attr.h>
#include "extract_interface.h"
#include "python.h"

static void die(const char *msg) __attribute__((noreturn));

/* Print error message "msg" and abort.
 */
static void die(const char *msg)
{
	fprintf(stderr, "%s", msg);
	abort();
}

/* Return a sequence of the types of which the given type declaration is
 * marked as being a subtype.
 * The order of the types is the opposite of the order in which they
 * appear in the source.  In particular, the first annotation
 * is the one that is closest to the annotated type and the corresponding
 * type is then also the first that will appear in the sequence of types.
 */
static vector<string> find_superclasses(RecordDecl *decl)
{
	vector<string> super;

	if (!decl->hasAttrs())
		return super;

	string sub = "isl_subclass";
	size_t len = sub.length();
	AttrVec attrs = decl->getAttrs();
	for (AttrVec::const_iterator i = attrs.begin() ; i != attrs.end(); ++i) {
		const AnnotateAttr *ann = dyn_cast<AnnotateAttr>(*i);
		if (!ann)
			continue;
		string s = ann->getAnnotation().str();
		if (s.substr(0, len) == sub) {
			s = s.substr(len + 1, s.length() - len  - 2);
			super.push_back(s);
		}
	}

	return super;
}

/* Is decl marked as being part of an overloaded method?
 */
static bool is_overload(Decl *decl)
{
	return has_annotation(decl, "isl_overload");
}

/* Is decl marked as a constructor?
 */
static bool is_constructor(Decl *decl)
{
	return has_annotation(decl, "isl_constructor");
}

/* Is decl marked as consuming a reference?
 */
static bool takes(Decl *decl)
{
	return has_annotation(decl, "isl_take");
}

/* Is decl marked as returning a reference that is required to be freed.
 */
static bool gives(Decl *decl)
{
	return has_annotation(decl, "isl_give");
}

/* isl_class collects all constructors and methods for an isl "class".
 * "name" is the name of the class.
 * "type" is the declaration that introduces the type.
 * "methods" contains the set of methods, grouped by method name.
 * "fn_to_str" is a reference to the *_to_str method of this class, if any.
 * "fn_free" is a reference to the *_free method of this class, if any.
 */
struct isl_class {
	string name;
	RecordDecl *type;
	set<FunctionDecl *> constructors;
	map<string, set<FunctionDecl *> > methods;
	FunctionDecl *fn_to_str;
	FunctionDecl *fn_free;

	bool is_static(FunctionDecl *method);

	void print(map<string, isl_class> &classes, set<string> &done);
	void print_constructor(FunctionDecl *method);
	void print_representation(const string &python_name);
	void print_method_type(FunctionDecl *fd);
	void print_method_types();
	void print_method(FunctionDecl *method, vector<string> super);
	void print_method_overload(FunctionDecl *method);
	void print_method(const string &fullname,
		const set<FunctionDecl *> &methods, vector<string> super);
};

/* Return the class that has a name that matches the initial part
 * of the name of function "fd" or NULL if no such class could be found.
 */
static isl_class *method2class(map<string, isl_class> &classes,
	FunctionDecl *fd)
{
	string best;
	map<string, isl_class>::iterator ci;
	string name = fd->getNameAsString();

	for (ci = classes.begin(); ci != classes.end(); ++ci) {
		if (name.substr(0, ci->first.length()) == ci->first)
			best = ci->first;
	}

	if (classes.find(best) == classes.end()) {
		cerr << "Unable to find class of " << name << endl;
		return NULL;
	}

	return &classes[best];
}

/* Is "type" the type "isl_ctx *"?
 */
static bool is_isl_ctx(QualType type)
{
	if (!type->isPointerType())
		return 0;
	type = type->getPointeeType();
	if (type.getAsString() != "isl_ctx")
		return false;

	return true;
}

/* Is the first argument of "fd" of type "isl_ctx *"?
 */
static bool first_arg_is_isl_ctx(FunctionDecl *fd)
{
	ParmVarDecl *param;

	if (fd->getNumParams() < 1)
		return false;

	param = fd->getParamDecl(0);
	return is_isl_ctx(param->getOriginalType());
}

/* Is "type" that of a pointer to an isl_* structure?
 */
static bool is_isl_type(QualType type)
{
	if (type->isPointerType()) {
		string s;

		type = type->getPointeeType();
		if (type->isFunctionType())
			return false;
		s = type.getAsString();
		return s.substr(0, 4) == "isl_";
	}

	return false;
}

/* Is "type" the type isl_bool?
 */
static bool is_isl_bool(QualType type)
{
	string s;

	if (type->isPointerType())
		return false;

	s = type.getAsString();
	return s == "isl_bool";
}

/* Is "type" that of a pointer to char.
 */
static bool is_string_type(QualType type)
{
	if (type->isPointerType()) {
		string s;

		type = type->getPointeeType();
		if (type->isFunctionType())
			return false;
		s = type.getAsString();
		return s == "const char" || "char";
	}

	return false;
}

/* Is "type" that of a pointer to a function?
 */
static bool is_callback(QualType type)
{
	if (!type->isPointerType())
		return false;
	type = type->getPointeeType();
	return type->isFunctionType();
}

/* Is "type" that of "char *" of "const char *"?
 */
static bool is_string(QualType type)
{
	if (type->isPointerType()) {
		string s = type->getPointeeType().getAsString();
		return s == "const char" || s == "char";
	}

	return false;
}

/* Return the name of the type that "type" points to.
 * The input "type" is assumed to be a pointer type.
 */
static string extract_type(QualType type)
{
	if (type->isPointerType())
		return type->getPointeeType().getAsString();
	die("Cannot extract type from non-pointer type");
}

/* Drop the "isl_" initial part of the type name "name".
 */
static string type2python(string name)
{
	return name.substr(4);
}

/* If "method" is overloaded, then drop the suffix of "name"
 * corresponding to the type of the final argument and
 * return the modified name (or the original name if
 * no modifications were made).
 */
static string drop_type_suffix(string name, FunctionDecl *method)
{
	int num_params;
	ParmVarDecl *param;
	string type;
	size_t name_len, type_len;

	if (!is_overload(method))
		return name;

	num_params = method->getNumParams();
	param = method->getParamDecl(num_params - 1);
	type = extract_type(param->getOriginalType());
	type = type.substr(4);
	name_len = name.length();
	type_len = type.length();

	if (name_len > type_len && name.substr(name_len - type_len) == type)
		name = name.substr(0, name_len - type_len - 1);

	return name;
}

/* Should "method" be considered to be a static method?
 * That is, is the first argument something other than
 * an instance of the class?
 */
bool isl_class::is_static(FunctionDecl *method)
{
	ParmVarDecl *param = method->getParamDecl(0);
	QualType type = param->getOriginalType();

	if (!is_isl_type(type))
		return true;
	return extract_type(type) != name;
}

/* Print the header of the method "name" with "n_arg" arguments.
 * If "is_static" is set, then mark the python method as static.
 *
 * If the method is called "from", then rename it to "convert_from"
 * because "from" is a python keyword.
 */
static void print_method_header(bool is_static, const string &name, int n_arg)
{
	const char *s;

	if (is_static)
		printf("    @staticmethod\n");

	s = name.c_str();
	if (name == "from")
		s = "convert_from";

	printf("    def %s(", s);
	for (int i = 0; i < n_arg; ++i) {
		if (i)
			printf(", ");
		printf("arg%d", i);
	}
	printf("):\n");
}

/* Print a check that the argument in position "pos" is of type "type".
 * If this fails and if "upcast" is set, then convert the first
 * argument to "super" and call the method "name" on it, passing
 * the remaining of the "n" arguments.
 * If the check fails and "upcast" is not set, then simply raise
 * an exception.
 * If "upcast" is not set, then the "super", "name" and "n" arguments
 * to this function are ignored.
 */
static void print_type_check(const string &type, int pos, bool upcast,
	const string &super, const string &name, int n)
{
	printf("        try:\n");
	printf("            if not arg%d.__class__ is %s:\n",
		pos, type.c_str());
	printf("                arg%d = %s(arg%d)\n",
		pos, type.c_str(), pos);
	printf("        except:\n");
	if (upcast) {
		printf("            return %s(arg0).%s(",
			type2python(super).c_str(), name.c_str());
		for (int i = 1; i < n; ++i) {
			if (i != 1)
				printf(", ");
			printf("arg%d", i);
		}
		printf(")\n");
	} else
		printf("            raise\n");
}

/* Construct a wrapper for a callback argument (at position "arg").
 * Assign the wrapper to "cb".  We assume here that a function call
 * has at most one callback argument.
 *
 * The wrapper converts the arguments of the callback to python types.
 * If any exception is thrown, the wrapper keeps track of it in exc_info[0]
 * and returns -1.  Otherwise the wrapper returns 0.
 */
static void print_callback(QualType type, int arg)
{
	const FunctionProtoType *fn = type->getAs<FunctionProtoType>();
	unsigned n_arg = fn->getNumArgs();

	printf("        exc_info = [None]\n");
	printf("        fn = CFUNCTYPE(c_int");
	for (unsigned i = 0; i < n_arg - 1; ++i) {
		if (!is_isl_type(fn->getArgType(i)))
			die("Argument has non-isl type");
		printf(", c_void_p");
	}
	printf(", c_void_p)\n");
	printf("        def cb_func(");
	for (unsigned i = 0; i < n_arg; ++i) {
		if (i)
			printf(", ");
		printf("cb_arg%d", i);
	}
	printf("):\n");
	for (unsigned i = 0; i < n_arg - 1; ++i) {
		string arg_type;
		arg_type = type2python(extract_type(fn->getArgType(i)));
		printf("            cb_arg%d = %s(ctx=arg0.ctx, "
			"ptr=cb_arg%d)\n", i, arg_type.c_str(), i);
	}
	printf("            try:\n");
	printf("                arg%d(", arg);
	for (unsigned i = 0; i < n_arg - 1; ++i) {
		if (i)
			printf(", ");
		printf("cb_arg%d", i);
	}
	printf(")\n");
	printf("            except:\n");
	printf("                import sys\n");
	printf("                exc_info[0] = sys.exc_info()\n");
	printf("                return -1\n");
	printf("            return 0\n");
	printf("        cb = fn(cb_func)\n");
}

/* Print the argument at position "arg" in call to "fd".
 * "skip" is the number of initial arguments of "fd" that are
 * skipped in the Python method.
 *
 * If the argument is a callback, then print a reference to
 * the callback wrapper "cb".
 * Otherwise, if the argument is marked as consuming a reference,
 * then pass a copy of the the pointer stored in the corresponding
 * argument passed to the Python method.
 * Otherwise, if the argument is a pointer, then pass this pointer itself.
 * Otherwise, pass the argument directly.
 */
static void print_arg_in_call(FunctionDecl *fd, int arg, int skip)
{
	ParmVarDecl *param = fd->getParamDecl(arg);
	QualType type = param->getOriginalType();
	if (is_callback(type)) {
		printf("cb");
	} else if (takes(param)) {
		string type_s = extract_type(type);
		printf("isl.%s_copy(arg%d.ptr)", type_s.c_str(), arg - skip);
	} else if (type->isPointerType()) {
		printf("arg%d.ptr", arg - skip);
	} else {
		printf("arg%d", arg - skip);
	}
}

/* Print the return statement of the python method corresponding
 * to the C function "method".
 *
 * If the return type is a (const) char *, then convert the result
 * to a Python string, raising an error on NULL and freeing
 * the C string if needed.
 *
 * If the return type is isl_bool, then convert the result to
 * a Python boolean, raising an error on isl_bool_error.
 */
static void print_method_return(FunctionDecl *method)
{
	QualType return_type = method->getReturnType();

	if (is_isl_type(return_type)) {
		string type;

		type = type2python(extract_type(return_type));
		printf("        return %s(ctx=ctx, ptr=res)\n", type.c_str());
	} else if (is_string_type(return_type)) {
		printf("        if res == 0:\n");
		printf("            raise\n");
		printf("        string = str(cast(res, c_char_p).value)\n");

		if (gives(method))
			printf("        libc.free(res)\n");

		printf("        return string\n");
	} else if (is_isl_bool(return_type)) {
		printf("        if res < 0:\n");
		printf("            raise\n");
		printf("        return bool(res)\n");
	} else {
		printf("        return res\n");
	}
}

/* Print a python method corresponding to the C function "method".
 * "super" contains the superclasses of the class to which the method belongs,
 * with the first element corresponding to the annotation that appears
 * closest to the annotated type.  This superclass is the least
 * general extension of the annotated type in the linearization
 * of the class hierarchy.
 *
 * If the first argument of "method" is something other than an instance
 * of the class, then mark the python method as static.
 * If, moreover, this first argument is an isl_ctx, then remove
 * it from the arguments of the Python method.
 *
 * If the function has a callback argument, then it also has a "user"
 * argument.  Since Python has closures, there is no need for such
 * a user argument in the Python interface, so we simply drop it.
 * We also create a wrapper ("cb") for the callback.
 *
 * For each argument of the function that refers to an isl structure,
 * including the object on which the method is called,
 * we check if the corresponding actual argument is of the right type.
 * If not, we try to convert it to the right type.
 * If that doesn't work and if "super" contains at least one element, we try
 * to convert self to the type of the first superclass in "super" and
 * call the corresponding method.
 *
 * If the function consumes a reference, then we pass it a copy of
 * the actual argument.
 */
void isl_class::print_method(FunctionDecl *method, vector<string> super)
{
	string fullname = method->getName();
	string cname = fullname.substr(name.length() + 1);
	int num_params = method->getNumParams();
	int drop_user = 0;
	int drop_ctx = first_arg_is_isl_ctx(method);

	for (int i = 1; i < num_params; ++i) {
		ParmVarDecl *param = method->getParamDecl(i);
		QualType type = param->getOriginalType();
		if (is_callback(type))
			drop_user = 1;
	}

	print_method_header(is_static(method), cname,
			    num_params - drop_ctx - drop_user);

	for (int i = drop_ctx; i < num_params; ++i) {
		ParmVarDecl *param = method->getParamDecl(i);
		string type;
		if (!is_isl_type(param->getOriginalType()))
			continue;
		type = type2python(extract_type(param->getOriginalType()));
		if (!drop_ctx && i > 0 && super.size() > 0)
			print_type_check(type, i - drop_ctx, true, super[0],
					cname, num_params - drop_user);
		else
			print_type_check(type, i - drop_ctx, false, "",
					cname, -1);
	}
	for (int i = 1; i < num_params; ++i) {
		ParmVarDecl *param = method->getParamDecl(i);
		QualType type = param->getOriginalType();
		if (!is_callback(type))
			continue;
		print_callback(type->getPointeeType(), i - drop_ctx);
	}
	if (drop_ctx)
		printf("        ctx = Context.getDefaultInstance()\n");
	else
		printf("        ctx = arg0.ctx\n");
	printf("        res = isl.%s(", fullname.c_str());
	if (drop_ctx)
		printf("ctx");
	else
		print_arg_in_call(method, 0, 0);
	for (int i = 1; i < num_params - drop_user; ++i) {
		printf(", ");
		print_arg_in_call(method, i, drop_ctx);
	}
	if (drop_user)
		printf(", None");
	printf(")\n");

	if (drop_user) {
		printf("        if exc_info[0] != None:\n");
		printf("            raise (exc_info[0][0], "
			"exc_info[0][1], exc_info[0][2])\n");
	}

	print_method_return(method);
}

/* Print part of an overloaded python method corresponding to the C function
 * "method".
 *
 * In particular, print code to test whether the arguments passed to
 * the python method correspond to the arguments expected by "method"
 * and to call "method" if they do.
 */
void isl_class::print_method_overload(FunctionDecl *method)
{
	string fullname = method->getName();
	int num_params = method->getNumParams();
	int first;
	string type;

	first = is_static(method) ? 0 : 1;

	printf("        if ");
	for (int i = first; i < num_params; ++i) {
		if (i > first)
			printf(" and ");
		ParmVarDecl *param = method->getParamDecl(i);
		if (is_isl_type(param->getOriginalType())) {
			string type;
			type = extract_type(param->getOriginalType());
			type = type2python(type);
			printf("arg%d.__class__ is %s", i, type.c_str());
		} else
			printf("type(arg%d) == str", i);
	}
	printf(":\n");
	printf("            res = isl.%s(", fullname.c_str());
	print_arg_in_call(method, 0, 0);
	for (int i = 1; i < num_params; ++i) {
		printf(", ");
		print_arg_in_call(method, i, 0);
	}
	printf(")\n");
	type = type2python(extract_type(method->getReturnType()));
	printf("            return %s(ctx=arg0.ctx, ptr=res)\n", type.c_str());
}

/* Print a python method with a name derived from "fullname"
 * corresponding to the C functions "methods".
 * "super" contains the superclasses of the class to which the method belongs.
 *
 * If "methods" consists of a single element that is not marked overloaded,
 * the use print_method to print the method.
 * Otherwise, print an overloaded method with pieces corresponding
 * to each function in "methods".
 */
void isl_class::print_method(const string &fullname,
	const set<FunctionDecl *> &methods, vector<string> super)
{
	string cname;
	set<FunctionDecl *>::const_iterator it;
	int num_params;
	FunctionDecl *any_method;

	any_method = *methods.begin();
	if (methods.size() == 1 && !is_overload(any_method)) {
		print_method(any_method, super);
		return;
	}

	cname = fullname.substr(name.length() + 1);
	num_params = any_method->getNumParams();

	print_method_header(is_static(any_method), cname, num_params);

	for (it = methods.begin(); it != methods.end(); ++it)
		print_method_overload(*it);
}

/* Print part of the constructor for this isl_class.
 *
 * In particular, check if the actual arguments correspond to the
 * formal arguments of "cons" and if so call "cons" and put the
 * result in self.ptr and a reference to the default context in self.ctx.
 *
 * If the function consumes a reference, then we pass it a copy of
 * the actual argument.
 */
void isl_class::print_constructor(FunctionDecl *cons)
{
	string fullname = cons->getName();
	string cname = fullname.substr(name.length() + 1);
	int num_params = cons->getNumParams();
	int drop_ctx = first_arg_is_isl_ctx(cons);

	printf("        if len(args) == %d", num_params - drop_ctx);
	for (int i = drop_ctx; i < num_params; ++i) {
		ParmVarDecl *param = cons->getParamDecl(i);
		QualType type = param->getOriginalType();
		if (is_isl_type(type)) {
			string s;
			s = type2python(extract_type(type));
			printf(" and args[%d].__class__ is %s",
				i - drop_ctx, s.c_str());
		} else if (type->isPointerType()) {
			printf(" and type(args[%d]) == str", i - drop_ctx);
		} else {
			printf(" and type(args[%d]) == int", i - drop_ctx);
		}
	}
	printf(":\n");
	printf("            self.ctx = Context.getDefaultInstance()\n");
	printf("            self.ptr = isl.%s(", fullname.c_str());
	if (drop_ctx)
		printf("self.ctx");
	for (int i = drop_ctx; i < num_params; ++i) {
		ParmVarDecl *param = cons->getParamDecl(i);
		if (i)
			printf(", ");
		if (is_isl_type(param->getOriginalType())) {
			if (takes(param)) {
				string type;
				type = extract_type(param->getOriginalType());
				printf("isl.%s_copy(args[%d].ptr)",
					type.c_str(), i - drop_ctx);
			} else
				printf("args[%d].ptr", i - drop_ctx);
		} else
			printf("args[%d]", i - drop_ctx);
	}
	printf(")\n");
	printf("            return\n");
}

/* Print the header of the class "name" with superclasses "super".
 * The order of the superclasses is the opposite of the order
 * in which the corresponding annotations appear in the source code.
 */
static void print_class_header(const string &name, const vector<string> &super)
{
	printf("class %s", name.c_str());
	if (super.size() > 0) {
		printf("(");
		for (unsigned i = 0; i < super.size(); ++i) {
			if (i > 0)
				printf(", ");
			printf("%s", type2python(super[i]).c_str());
		}
		printf(")");
	} else {
		printf("(object)");
	}
	printf(":\n");
}

/* Tell ctypes about the return type of "fd".
 * In particular, if "fd" returns a pointer to an isl object,
 * then tell ctypes it returns a "c_void_p".
 * Similarly, if "fd" returns an isl_bool,
 * then tell ctypes it returns a "c_bool".
 * If "fd" returns a char *, then simply tell ctypes.
 */
static void print_restype(FunctionDecl *fd)
{
	string fullname = fd->getName();
	QualType type = fd->getReturnType();
	if (is_isl_type(type))
		printf("isl.%s.restype = c_void_p\n", fullname.c_str());
	else if (is_isl_bool(type))
		printf("isl.%s.restype = c_bool\n", fullname.c_str());
	else if (is_string_type(type))
		printf("isl.%s.restype = POINTER(c_char)\n", fullname.c_str());
}

/* Tell ctypes about the types of the arguments of the function "fd".
 */
static void print_argtypes(FunctionDecl *fd)
{
	string fullname = fd->getName();
	int n = fd->getNumParams();
	int drop_user = 0;

	printf("isl.%s.argtypes = [", fullname.c_str());
	for (int i = 0; i < n - drop_user; ++i) {
		ParmVarDecl *param = fd->getParamDecl(i);
		QualType type = param->getOriginalType();
		if (is_callback(type))
			drop_user = 1;
		if (i)
			printf(", ");
		if (is_isl_ctx(type))
			printf("Context");
		else if (is_isl_type(type) || is_callback(type))
			printf("c_void_p");
		else if (is_string(type))
			printf("c_char_p");
		else
			printf("c_int");
	}
	if (drop_user)
		printf(", c_void_p");
	printf("]\n");
}

/* Print type definitions for the method 'fd'.
 */
void isl_class::print_method_type(FunctionDecl *fd)
{
	print_restype(fd);
	print_argtypes(fd);
}

/* Print declarations for methods printing the class representation,
 * provided there is a corresponding *_to_str function.
 *
 * In particular, provide an implementation of __str__ and __repr__ methods to
 * override the default representation used by python. Python uses __str__ to
 * pretty print the class (e.g., when calling print(obj)) and uses __repr__
 * when printing a precise representation of an object (e.g., when dumping it
 * in the REPL console).
 *
 * Check the type of the argument before calling the *_to_str function
 * on it in case the method was called on an object from a subclass.
 */
void isl_class::print_representation(const string &python_name)
{
	if (!fn_to_str)
		return;

	printf("    def __str__(arg0):\n");
	print_type_check(python_name, 0, false, "", "", -1);
	printf("        ptr = isl.%s(arg0.ptr)\n",
		string(fn_to_str->getName()).c_str());
	printf("        res = str(cast(ptr, c_char_p).value)\n");
	printf("        libc.free(ptr)\n");
	printf("        return res\n");
	printf("    def __repr__(self):\n");
	printf("        s = str(self)\n");
	printf("        if '\"' in s:\n");
	printf("            return 'isl.%s(\"\"\"%%s\"\"\")' %% s\n",
		python_name.c_str());
	printf("        else:\n");
	printf("            return 'isl.%s(\"%%s\")' %% s\n",
		python_name.c_str());
}

/* Print code to set method type signatures.
 *
 * To be able to call C functions it is necessary to explicitly set their
 * argument and result types.  Do this for all exported constructors and
 * methods, as well as for the *_to_str method, if it exists.
 * Assuming each exported class has a *_free method,
 * also unconditionally set the type of such methods.
 */
void isl_class::print_method_types()
{
	set<FunctionDecl *>::iterator in;
	map<string, set<FunctionDecl *> >::iterator it;

	for (in = constructors.begin(); in != constructors.end(); ++in)
		print_method_type(*in);

	for (it = methods.begin(); it != methods.end(); ++it)
		for (in = it->second.begin(); in != it->second.end(); ++in)
			print_method_type(*in);

	print_method_type(fn_free);
	if (fn_to_str)
		print_method_type(fn_to_str);
}

/* Print out the definition of this isl_class.
 *
 * We first check if this isl_class is a subclass of one or more other classes.
 * If it is, we make sure those superclasses are printed out first.
 *
 * Then we print a constructor with several cases, one for constructing
 * a Python object from a return value and one for each function that
 * was marked as a constructor.
 *
 * Next, we print out some common methods and the methods corresponding
 * to functions that are not marked as constructors.
 *
 * Finally, we tell ctypes about the types of the arguments of the
 * constructor functions and the return types of those function returning
 * an isl object.
 */
void isl_class::print(map<string, isl_class> &classes, set<string> &done)
{
	string p_name = type2python(name);
	set<FunctionDecl *>::iterator in;
	map<string, set<FunctionDecl *> >::iterator it;
	vector<string> super = find_superclasses(type);

	for (unsigned i = 0; i < super.size(); ++i)
		if (done.find(super[i]) == done.end())
			classes[super[i]].print(classes, done);
	done.insert(name);

	printf("\n");
	print_class_header(p_name, super);
	printf("    def __init__(self, *args, **keywords):\n");

	printf("        if \"ptr\" in keywords:\n");
	printf("            self.ctx = keywords[\"ctx\"]\n");
	printf("            self.ptr = keywords[\"ptr\"]\n");
	printf("            return\n");

	for (in = constructors.begin(); in != constructors.end(); ++in)
		print_constructor(*in);
	printf("        raise Error\n");
	printf("    def __del__(self):\n");
	printf("        if hasattr(self, 'ptr'):\n");
	printf("            isl.%s_free(self.ptr)\n", name.c_str());

	print_representation(p_name);

	for (it = methods.begin(); it != methods.end(); ++it)
		print_method(it->first, it->second, super);

	printf("\n");

	print_method_types();
}

/* Generate a python interface based on the extracted types and functions.
 * We first collect all functions that belong to a certain type,
 * separating constructors from regular methods and keeping track
 * of the _to_str and _free functions, if any, separately.  If there are any
 * overloaded functions, then they are grouped based on their name
 * after removing the argument type suffix.
 *
 * Then we print out each class in turn.  If one of these is a subclass
 * of some other class, it will make sure the superclass is printed out first.
 */
void generate_python(set<RecordDecl *> &exported_types,
	set<FunctionDecl *> exported_functions, set<FunctionDecl *> functions)
{
	map<string, isl_class> classes;
	map<string, isl_class>::iterator ci;
	set<string> done;
	map<string, FunctionDecl *> functions_by_name;

	set<FunctionDecl *>::iterator in;
	for (in = functions.begin(); in != functions.end(); ++in) {
		FunctionDecl *decl = *in;
		functions_by_name[decl->getName()] = decl;
	}

	set<RecordDecl *>::iterator it;
	for (it = exported_types.begin(); it != exported_types.end(); ++it) {
		RecordDecl *decl = *it;
		map<string, FunctionDecl *>::iterator i;

		string name = decl->getName();
		classes[name].name = name;
		classes[name].type = decl;
		classes[name].fn_to_str = NULL;
		classes[name].fn_free = NULL;

		i = functions_by_name.find(name + "_to_str");
		if (i != functions_by_name.end())
			classes[name].fn_to_str = i->second;

		i = functions_by_name.find (name + "_free");
		if (i == functions_by_name.end())
			die("No _free function found");
		classes[name].fn_free = i->second;
	}

	for (in = exported_functions.begin(); in != exported_functions.end();
	     ++in) {
		isl_class *c = method2class(classes, *in);
		if (!c)
			continue;
		if (is_constructor(*in)) {
			c->constructors.insert(*in);
		} else {
			FunctionDecl *method = *in;
			string fullname = method->getName();
			fullname = drop_type_suffix(fullname, method);
			c->methods[fullname].insert(method);
		}
	}

	for (ci = classes.begin(); ci != classes.end(); ++ci) {
		if (done.find(ci->first) == done.end())
			ci->second.print(classes, done);
	}
}
