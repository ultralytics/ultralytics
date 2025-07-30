#ifndef ISL_LOCAL_H
#define ISL_LOCAL_H

#include <isl/mat.h>

isl_bool isl_local_div_is_known(__isl_keep isl_mat *div, int pos);
int isl_local_cmp(__isl_keep isl_mat *div1, __isl_keep isl_mat *div2);

#endif
