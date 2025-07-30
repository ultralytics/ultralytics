AC_DEFUN([AX_DETECT_GIT_HEAD], [
	AC_SUBST(GIT_HEAD_ID)
	AC_SUBST(GIT_HEAD)
	AC_SUBST(GIT_HEAD_VERSION)
	if test -f $srcdir/.git; then
		gitdir=`GIT_DIR=$srcdir/.git git rev-parse --git-dir`
		GIT_HEAD="$gitdir/index"
		GIT_REPO="$gitdir"
		GIT_HEAD_ID=`GIT_DIR=$GIT_REPO git describe --always`
	elif test -f $srcdir/.git/HEAD; then
		GIT_HEAD="$srcdir/.git/index"
		GIT_REPO="$srcdir/.git"
		GIT_HEAD_ID=`GIT_DIR=$GIT_REPO git describe --always`
	elif test -f $srcdir/GIT_HEAD_ID; then
		GIT_HEAD_ID=`cat $srcdir/GIT_HEAD_ID`
	else
		mysrcdir=`(cd $srcdir; pwd)`
		head=`basename $mysrcdir | sed -e 's/.*-//'`
		head2=`echo $head | sed -e 's/[^0-9a-f]//'`
		head3=`echo $head2 | sed -e 's/........................................//'`
		if test "x$head3" = "x" -a "x$head" = "x$head2"; then
			GIT_HEAD_ID="$head"
		else
			GIT_HEAD_ID="UNKNOWN"
		fi
	fi
	if test -z "$GIT_REPO" ; then
		GIT_HEAD_VERSION="$GIT_HEAD_ID"
	else
	    GIT_HEAD_VERSION="\`GIT_DIR=$GIT_REPO git describe --always\`"
	fi
])
