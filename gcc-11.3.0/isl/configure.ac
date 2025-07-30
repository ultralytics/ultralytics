AC_INIT([isl], [0.18], [isl-development@googlegroups.com])
AC_CONFIG_AUX_DIR([.])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign])
m4_ifdef([AM_SILENT_RULES],[AM_SILENT_RULES([yes])])
AC_SUBST(versioninfo)
versioninfo=18:0:3

if test "x$prefix" != "xNONE"; then
	prefix_wd=`cd $prefix && pwd`
	srcdir_wd=`cd $srcdir && pwd`
	wd=`pwd`
	if test "x$prefix_wd" = "x$srcdir_wd"; then
		AC_MSG_ERROR(Installation in source directory not supported)
	fi
	if test "x$prefix_wd" = "x$wd"; then
		AC_MSG_ERROR(Installation in build directory not supported)
	fi
fi

AC_PROG_CC
AC_PROG_CXX

AX_CC_MAXOPT
AX_GCC_WARN_UNUSED_RESULT
AX_C___ATTRIBUTE__

AC_PROG_LIBTOOL

AC_CHECK_PROG(PERL, perl, perl, [])
AC_CHECK_PROG(PDFLATEX, pdflatex, pdflatex, [])
AC_CHECK_PROG(POD2HTML, pod2html, pod2html, [])

AM_CONDITIONAL(GENERATE_DOC, test -n "$PERL" -a -n "$PDFLATEX" -a -n "$POD2HTML")

AX_CREATE_STDINT_H(include/isl/stdint.h)

AC_ARG_WITH([int],
	    [AS_HELP_STRING([--with-int=gmp|imath|imath-32],
			    [Which package to use to represent
				multi-precision integers [default=gmp]])],
	    [], [with_int=gmp])
case "$with_int" in
gmp|imath|imath-32)
	;;
*)
	AC_MSG_ERROR(
	    [bad value ${withval} for --with-int (use gmp, imath or imath-32)])
esac

AC_SUBST(MP_CPPFLAGS)
AC_SUBST(MP_LDFLAGS)
AC_SUBST(MP_LIBS)
case "$with_int" in
gmp)
	AX_DETECT_GMP
	;;
imath|imath-32)
	AX_DETECT_IMATH
	;;
esac
if test "x$with_int" = "ximath-32" -a "x$GCC" = "xyes"; then
	MP_CPPFLAGS="-std=gnu99 $MP_CPPFLAGS"
fi

AM_CONDITIONAL(IMATH_FOR_MP, test x$with_int = ximath -o x$with_int = ximath-32)
AM_CONDITIONAL(GMP_FOR_MP, test x$with_int = xgmp)

AM_CONDITIONAL(SMALL_INT_OPT, test "x$with_int" == "ximath-32")
AS_IF([test "x$with_int" == "ximath-32"], [
	AC_DEFINE([USE_SMALL_INT_OPT], [], [Use small integer optimization])
])

AC_CHECK_DECLS(ffs,[],[],[#include <strings.h>])
AC_CHECK_DECLS(__builtin_ffs,[],[],[])
AC_CHECK_DECLS([_BitScanForward],[],[],[#include <intrin.h>])
if test "x$ac_cv_have_decl_ffs" = xno -a \
		"x$ac_cv_have_decl___builtin_ffs" = xno -a \
		"x$ac_cv_have_decl__BitScanForward" = xno; then
	AC_MSG_ERROR([No ffs implementation found])
fi
AC_CHECK_DECLS([strcasecmp,strncasecmp],[],[],[#include <strings.h>])
AC_CHECK_DECLS([_stricmp,_strnicmp],[],[],[#include <string.h>])
if test "x$ac_cv_have_decl_strcasecmp" = xno -a \
		"x$ac_cv_have_decl__stricmp" = xno; then
	AC_MSG_ERROR([No strcasecmp implementation found])
fi
if test "x$ac_cv_have_decl_strncasecmp" = xno -a \
		"x$ac_cv_have_decl__strnicmp" = xno; then
	AC_MSG_ERROR([No strncasecmp implementation found])
fi
AC_CHECK_DECLS([snprintf,_snprintf],[],[],[#include <stdio.h>])
if test "x$ac_cv_have_decl_snprintf" = xno -a \
		"x$ac_cv_have_decl__snprintf" = xno; then
	AC_MSG_ERROR([No snprintf implementation found])
fi

AC_SUBST(CLANG_CXXFLAGS)
AC_SUBST(CLANG_LDFLAGS)
AC_SUBST(CLANG_LIBS)
AX_SUBMODULE(clang,system|no,no)
case "$with_clang" in
system)
	AC_PROG_GREP
	AC_PROG_SED
	llvm_config="llvm-config"
	AC_CHECK_PROG([llvm_config_found], ["$llvm_config"], [yes])
	if test "x$with_clang_prefix" != "x"; then
		llvm_config="$with_clang_prefix/bin/llvm-config"
		if test -x "$llvm_config"; then
			llvm_config_found=yes
		fi
	fi
	if test "$llvm_config_found" != yes; then
		AC_MSG_ERROR([llvm-config not found])
	fi
	CLANG_CXXFLAGS=`$llvm_config --cxxflags | \
		$SED -e 's/-Wcovered-switch-default//'`
	CLANG_LDFLAGS=`$llvm_config --ldflags`
	targets=`$llvm_config --targets-built`
	components="$targets asmparser bitreader support mc"
	$llvm_config --components | $GREP option > /dev/null 2> /dev/null
	if test $? -eq 0; then
		components="$components option"
	fi
	CLANG_LIBS=`$llvm_config --libs $components`
	systemlibs=`$llvm_config --system-libs 2> /dev/null | tail -1`
	if test $? -eq 0; then
		CLANG_LIBS="$CLANG_LIBS $systemlibs"
	fi
	CLANG_PREFIX=`$llvm_config --prefix`
	AC_DEFINE_UNQUOTED(CLANG_PREFIX, ["$CLANG_PREFIX"],
				[Clang installation prefix])

	SAVE_CPPFLAGS="$CPPFLAGS"
	CPPFLAGS="$CLANG_CXXFLAGS $CPPFLAGS"
	AC_LANG_PUSH(C++)
	AC_CHECK_HEADER([clang/Basic/SourceLocation.h], [],
		[AC_ERROR([clang header file not found])])
	AC_EGREP_HEADER([getDefaultTargetTriple], [llvm/Support/Host.h], [],
		[AC_DEFINE([getDefaultTargetTriple], [getHostTriple],
		[Define to getHostTriple for older versions of clang])])
	AC_EGREP_HEADER([getExpansionLineNumber],
		[clang/Basic/SourceLocation.h], [],
		[AC_DEFINE([getExpansionLineNumber],
			[getInstantiationLineNumber],
	    [Define to getInstantiationLineNumber for older versions of clang])])
	AC_EGREP_HEADER([DiagnosticsEngine],
		[clang/Basic/Diagnostic.h], [],
		[AC_DEFINE([DiagnosticsEngine],
			[Diagnostic],
	    [Define to Diagnostic for older versions of clang])])
	AC_EGREP_HEADER([ArrayRef], [clang/Driver/Driver.h],
		[AC_DEFINE([USE_ARRAYREF], [],
			[Define if Driver::BuildCompilation takes ArrayRef])])
	AC_EGREP_HEADER([CXXIsProduction], [clang/Driver/Driver.h],
		[AC_DEFINE([HAVE_CXXISPRODUCTION], [],
		[Define if Driver constructor takes CXXIsProduction argument])])
	AC_EGREP_HEADER([ IsProduction], [clang/Driver/Driver.h],
		[AC_DEFINE([HAVE_ISPRODUCTION], [],
		[Define if Driver constructor takes IsProduction argument])])
	AC_TRY_COMPILE([#include <clang/Driver/Driver.h>], [
		using namespace clang;
		DiagnosticsEngine *Diags;
		new driver::Driver("", "", "", *Diags);
	], [AC_DEFINE([DRIVER_CTOR_TAKES_DEFAULTIMAGENAME], [],
		      [Define if Driver constructor takes default image name])])
	AC_EGREP_HEADER([void HandleTopLevelDecl\(], [clang/AST/ASTConsumer.h],
		[AC_DEFINE([HandleTopLevelDeclReturn], [void],
			   [Return type of HandleTopLevelDeclReturn])
		 AC_DEFINE([HandleTopLevelDeclContinue], [],
			   [Return type of HandleTopLevelDeclReturn])],
		[AC_DEFINE([HandleTopLevelDeclReturn], [bool],
			   [Return type of HandleTopLevelDeclReturn])
		 AC_DEFINE([HandleTopLevelDeclContinue], [true],
			   [Return type of HandleTopLevelDeclReturn])])
	AC_CHECK_HEADER([clang/Basic/DiagnosticOptions.h],
		[AC_DEFINE([HAVE_BASIC_DIAGNOSTICOPTIONS_H], [],
			   [Define if clang/Basic/DiagnosticOptions.h exists])])
	AC_CHECK_HEADER([clang/Lex/PreprocessorOptions.h],
		[AC_DEFINE([HAVE_LEX_PREPROCESSOROPTIONS_H], [],
			   [Define if clang/Lex/PreprocessorOptions.h exists])],
		[], [#include <clang/Basic/LLVM.h>])
	AC_TRY_COMPILE([#include <clang/Basic/TargetInfo.h>], [
		using namespace clang;
		std::shared_ptr<TargetOptions> TO;
		DiagnosticsEngine *Diags;
		TargetInfo::CreateTargetInfo(*Diags, TO);
	], [AC_DEFINE([CREATETARGETINFO_TAKES_SHARED_PTR], [],
		    [Define if TargetInfo::CreateTargetInfo takes shared_ptr])])
	AC_TRY_COMPILE([#include <clang/Basic/TargetInfo.h>], [
		using namespace clang;
		TargetOptions *TO;
		DiagnosticsEngine *Diags;
		TargetInfo::CreateTargetInfo(*Diags, TO);
	], [AC_DEFINE([CREATETARGETINFO_TAKES_POINTER], [],
		      [Define if TargetInfo::CreateTargetInfo takes pointer])])
	AC_TRY_COMPILE([#include <clang/Frontend/CompilerInstance.h>], [
		using namespace clang;
		DiagnosticConsumer *client;
		CompilerInstance *Clang;
		Clang->createDiagnostics(client);
	], [], [AC_DEFINE([CREATEDIAGNOSTICS_TAKES_ARG], [],
	[Define if CompilerInstance::createDiagnostics takes argc and argv])])
	AC_TRY_COMPILE([#include <clang/Lex/HeaderSearchOptions.h>], [
		using namespace clang;
		HeaderSearchOptions HSO;
		HSO.AddPath("", frontend::Angled, false, false);
	], [AC_DEFINE([ADDPATH_TAKES_4_ARGUMENTS], [],
		[Define if HeaderSearchOptions::AddPath takes 4 arguments])])
	AC_EGREP_HEADER([getNumParams],
		[clang/AST/CanonicalType.h],
		[AC_DEFINE([getNumArgs], [getNumParams],
		    [Define to getNumParams for newer versions of clang])
		 AC_DEFINE([getArgType], [getParamType],
		    [Define to getParamType for newer versions of clang])])
	AC_EGREP_HEADER([getReturnType],
		[clang/AST/CanonicalType.h], [],
		[AC_DEFINE([getReturnType], [getResultType],
		    [Define to getResultType for older versions of clang])])
	AC_TRY_COMPILE([#include <clang/Frontend/CompilerInstance.h>], [
		using namespace clang;
		CompilerInstance *Clang;
		Clang->createPreprocessor(TU_Complete);
	], [AC_DEFINE([CREATEPREPROCESSOR_TAKES_TUKIND], [],
	[Define if CompilerInstance::createPreprocessor takes
	 TranslationUnitKind])])
	AC_EGREP_HEADER([setMainFileID], [clang/Basic/SourceManager.h],
		[AC_DEFINE([HAVE_SETMAINFILEID], [],
		[Define if SourceManager has a setMainFileID method])])
	AC_CHECK_HEADER([llvm/ADT/OwningPtr.h],
		[AC_DEFINE([HAVE_ADT_OWNINGPTR_H], [],
			   [Define if llvm/ADT/OwningPtr.h exists])])
	AC_EGREP_HEADER([initializeBuiltins],
		[clang/Basic/Builtins.h], [],
		[AC_DEFINE([initializeBuiltins], [InitializeBuiltins],
		[Define to InitializeBuiltins for older versions of clang])])
	AC_TRY_COMPILE([
		#include <clang/Basic/TargetOptions.h>
		#include <clang/Lex/PreprocessorOptions.h>
		#include <clang/Frontend/CompilerInstance.h>
	], [
		using namespace clang;
		CompilerInstance *Clang;
		TargetOptions TO;
		llvm::Triple T(TO.Triple);
		PreprocessorOptions PO;
		CompilerInvocation::setLangDefaults(Clang->getLangOpts(), IK_C,
				T, PO, LangStandard::lang_unspecified);
	], [AC_DEFINE([SETLANGDEFAULTS_TAKES_5_ARGUMENTS], [],
	    [Define if CompilerInvocation::setLangDefaults takes 5 arguments])])
	AC_LANG_POP
	CPPFLAGS="$SAVE_CPPFLAGS"

	SAVE_LDFLAGS="$LDFLAGS"
	LDFLAGS="$CLANG_LDFLAGS $LDFLAGS"
	AC_SUBST(LIB_CLANG_EDIT)
	AC_CHECK_LIB([clangEdit], [main], [LIB_CLANG_EDIT=-lclangEdit], [])
	LDFLAGS="$SAVE_LDFLAGS"
	;;
esac
AM_CONDITIONAL(HAVE_CLANG, test $with_clang = system)

AX_SET_WARNING_FLAGS

AC_SUBST(WARNING_FLAGS)

PACKAGE_CFLAGS="$MP_CPPFLAGS"
PACKAGE_LDFLAGS="$MP_LDFLAGS"
PACKAGE_LIBS="-lisl $MP_LIBS"
AX_CREATE_PKGCONFIG_INFO

AX_DETECT_GIT_HEAD

AH_BOTTOM([#include <isl_config_post.h>])
AC_CONFIG_HEADERS(isl_config.h)
AC_CONFIG_FILES(isl_srcdir.c)
AC_CONFIG_FILES(Makefile)
AC_CONFIG_FILES(doc/Makefile)
if test $with_clang = system; then
	AC_CONFIG_FILES(interface/Makefile)
fi
AC_CONFIG_FILES([bound_test.sh], [chmod +x bound_test.sh])
AC_CONFIG_FILES([codegen_test.sh], [chmod +x codegen_test.sh])
AC_CONFIG_FILES([pip_test.sh], [chmod +x pip_test.sh])
AC_CONFIG_COMMANDS_POST([
	dnl pass on arguments to subdir configures, but don't
	dnl add them to config.status
	ac_configure_args="$ac_configure_args $isl_configure_args"
])
AC_OUTPUT
