AC_DEFUN([AX_SUBMODULE],
[

m4_if(m4_bregexp($2,|,choice),choice,
	[AC_ARG_WITH($1,
		[AS_HELP_STRING([--with-$1=$2],
				[Which $1 to use [default=$3]])])])
case "system" in
$2)
	AC_ARG_WITH($1_prefix,
		    [AS_HELP_STRING([--with-$1-prefix=DIR],
				    [Prefix of $1 installation])])
	AC_ARG_WITH($1_exec_prefix,
		    [AS_HELP_STRING([--with-$1-exec-prefix=DIR],
				    [Exec prefix of $1 installation])])
esac
m4_if(m4_bregexp($2,build,build),build,
	[AC_ARG_WITH($1_builddir,
		[AS_HELP_STRING([--with-$1-builddir=DIR],
				[Location of $1 builddir])])])
if test "x$with_$1_prefix" != "x" -a "x$with_$1_exec_prefix" = "x"; then
	with_$1_exec_prefix=$with_$1_prefix
fi
if test "x$with_$1_prefix" != "x" -o "x$with_$1_exec_prefix" != "x"; then
	if test "x$with_$1" != "x" -a "x$with_$1" != "xyes" -a "x$with_$1" != "xsystem"; then
		AC_MSG_ERROR([Setting $with_$1_prefix implies use of system $1])
	fi
	with_$1="system"
fi
if test "x$with_$1_builddir" != "x"; then
	if test "x$with_$1" != "x" -a "x$with_$1" != "xyes" -a "x$with_$1" != "xbuild"; then
		AC_MSG_ERROR([Setting $with_$1_builddir implies use of build $1])
	fi
	with_$1="build"
	$1_srcdir=`echo @abs_srcdir@ | $with_$1_builddir/config.status --file=-`
	AC_MSG_NOTICE($1 sources in $$1_srcdir)
fi
if test "x$with_$1_exec_prefix" != "x"; then
	export PKG_CONFIG_PATH="$with_$1_exec_prefix/lib/pkgconfig${PKG_CONFIG_PATH+:$PKG_CONFIG_PATH}"
fi
case "$with_$1" in
$2)
	;;
*)
	case "$3" in
	bundled)
		if test -d $srcdir/.git -a \
			-d $srcdir/$1 -a \
			! -d $srcdir/$1/.git; then
			AC_MSG_WARN([git repo detected, but submodule $1 not initialized])
			AC_MSG_WARN([You may want to run])
			AC_MSG_WARN([	git submodule init])
			AC_MSG_WARN([	git submodule update])
			AC_MSG_WARN([	sh autogen.sh])
		fi
		if test -f $srcdir/$1/configure; then
			with_$1="bundled"
		else
			with_$1="no"
		fi
		;;
	*)
		with_$1="$3"
		;;
	esac
	;;
esac
AC_MSG_CHECKING([which $1 to use])
AC_MSG_RESULT($with_$1)

])
