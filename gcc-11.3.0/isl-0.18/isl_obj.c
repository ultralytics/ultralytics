/*
 * Copyright 2010      INRIA Saclay
 * Copyright 2014      Ecole Normale Superieure
 * Copyright 2014      INRIA Rocquencourt
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 * and Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 */

#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/polynomial.h>
#include <isl/schedule.h>
#include <isl/obj.h>

static void *isl_obj_val_copy(void *v)
{
	return isl_val_copy((isl_val *)v);
}

static void isl_obj_val_free(void *v)
{
	isl_val_free((isl_val *)v);
}

static __isl_give isl_printer *isl_obj_val_print(__isl_take isl_printer *p,
	void *v)
{
	return isl_printer_print_val(p, (isl_val *)v);
}

static void *isl_obj_val_add(void *v1, void *v2)
{
	return isl_val_add((isl_val *) v1, (isl_val *) v2);
}

struct isl_obj_vtable isl_obj_val_vtable = {
	isl_obj_val_copy,
	isl_obj_val_add,
	isl_obj_val_print,
	isl_obj_val_free
};

static void *isl_obj_map_copy(void *v)
{
	return isl_map_copy((struct isl_map *)v);
}

static void isl_obj_map_free(void *v)
{
	isl_map_free((struct isl_map *)v);
}

static __isl_give isl_printer *isl_obj_map_print(__isl_take isl_printer *p,
	void *v)
{
	return isl_printer_print_map(p, (struct isl_map *)v);
}

static void *isl_obj_map_add(void *v1, void *v2)
{
	return isl_map_union((struct isl_map *)v1, (struct isl_map *)v2);
}

struct isl_obj_vtable isl_obj_map_vtable = {
	isl_obj_map_copy,
	isl_obj_map_add,
	isl_obj_map_print,
	isl_obj_map_free
};

static void *isl_obj_union_map_copy(void *v)
{
	return isl_union_map_copy((isl_union_map *)v);
}

static void isl_obj_union_map_free(void *v)
{
	isl_union_map_free((isl_union_map *)v);
}

static __isl_give isl_printer *isl_obj_union_map_print(__isl_take isl_printer *p,
	void *v)
{
	return isl_printer_print_union_map(p, (isl_union_map *)v);
}

static void *isl_obj_union_map_add(void *v1, void *v2)
{
	return isl_union_map_union((isl_union_map *)v1, (isl_union_map *)v2);
}

struct isl_obj_vtable isl_obj_union_map_vtable = {
	isl_obj_union_map_copy,
	isl_obj_union_map_add,
	isl_obj_union_map_print,
	isl_obj_union_map_free
};

static void *isl_obj_set_copy(void *v)
{
	return isl_set_copy((struct isl_set *)v);
}

static void isl_obj_set_free(void *v)
{
	isl_set_free((struct isl_set *)v);
}

static __isl_give isl_printer *isl_obj_set_print(__isl_take isl_printer *p,
	void *v)
{
	return isl_printer_print_set(p, (struct isl_set *)v);
}

static void *isl_obj_set_add(void *v1, void *v2)
{
	return isl_set_union((struct isl_set *)v1, (struct isl_set *)v2);
}

struct isl_obj_vtable isl_obj_set_vtable = {
	isl_obj_set_copy,
	isl_obj_set_add,
	isl_obj_set_print,
	isl_obj_set_free
};

static void *isl_obj_union_set_copy(void *v)
{
	return isl_union_set_copy((isl_union_set *)v);
}

static void isl_obj_union_set_free(void *v)
{
	isl_union_set_free((isl_union_set *)v);
}

static __isl_give isl_printer *isl_obj_union_set_print(__isl_take isl_printer *p,
	void *v)
{
	return isl_printer_print_union_set(p, (isl_union_set *)v);
}

static void *isl_obj_union_set_add(void *v1, void *v2)
{
	return isl_union_set_union((isl_union_set *)v1, (isl_union_set *)v2);
}

struct isl_obj_vtable isl_obj_union_set_vtable = {
	isl_obj_union_set_copy,
	isl_obj_union_set_add,
	isl_obj_union_set_print,
	isl_obj_union_set_free
};

static void *isl_obj_pw_multi_aff_copy(void *v)
{
	return isl_pw_multi_aff_copy((isl_pw_multi_aff *) v);
}

static void isl_obj_pw_multi_aff_free(void *v)
{
	isl_pw_multi_aff_free((isl_pw_multi_aff *) v);
}

static __isl_give isl_printer *isl_obj_pw_multi_aff_print(
	__isl_take isl_printer *p, void *v)
{
	return isl_printer_print_pw_multi_aff(p, (isl_pw_multi_aff *) v);
}

static void *isl_obj_pw_multi_aff_add(void *v1, void *v2)
{
	return isl_pw_multi_aff_add((isl_pw_multi_aff *) v1,
				    (isl_pw_multi_aff *) v2);
}

struct isl_obj_vtable isl_obj_pw_multi_aff_vtable = {
	isl_obj_pw_multi_aff_copy,
	isl_obj_pw_multi_aff_add,
	isl_obj_pw_multi_aff_print,
	isl_obj_pw_multi_aff_free
};

static void *isl_obj_none_copy(void *v)
{
	return v;
}

static void isl_obj_none_free(void *v)
{
}

static __isl_give isl_printer *isl_obj_none_print(__isl_take isl_printer *p,
	void *v)
{
	return p;
}

static void *isl_obj_none_add(void *v1, void *v2)
{
	return NULL;
}

struct isl_obj_vtable isl_obj_none_vtable = {
	isl_obj_none_copy,
	isl_obj_none_add,
	isl_obj_none_print,
	isl_obj_none_free
};

static void *isl_obj_pw_qp_copy(void *v)
{
	return isl_pw_qpolynomial_copy((struct isl_pw_qpolynomial *)v);
}

static void isl_obj_pw_qp_free(void *v)
{
	isl_pw_qpolynomial_free((struct isl_pw_qpolynomial *)v);
}

static __isl_give isl_printer *isl_obj_pw_qp_print(__isl_take isl_printer *p,
	void *v)
{
	return isl_printer_print_pw_qpolynomial(p,
						(struct isl_pw_qpolynomial *)v);
}

static void *isl_obj_pw_qp_add(void *v1, void *v2)
{
	return isl_pw_qpolynomial_add((struct isl_pw_qpolynomial *)v1,
					(struct isl_pw_qpolynomial *)v2);
}

struct isl_obj_vtable isl_obj_pw_qpolynomial_vtable = {
	isl_obj_pw_qp_copy,
	isl_obj_pw_qp_add,
	isl_obj_pw_qp_print,
	isl_obj_pw_qp_free
};

static void *isl_obj_union_pw_qp_copy(void *v)
{
	return isl_union_pw_qpolynomial_copy((struct isl_union_pw_qpolynomial *)v);
}

static void isl_obj_union_pw_qp_free(void *v)
{
	isl_union_pw_qpolynomial_free((struct isl_union_pw_qpolynomial *)v);
}

static __isl_give isl_printer *isl_obj_union_pw_qp_print(
	__isl_take isl_printer *p, void *v)
{
	return isl_printer_print_union_pw_qpolynomial(p,
					(struct isl_union_pw_qpolynomial *)v);
}

static void *isl_obj_union_pw_qp_add(void *v1, void *v2)
{
	return isl_union_pw_qpolynomial_add(
					(struct isl_union_pw_qpolynomial *)v1,
					(struct isl_union_pw_qpolynomial *)v2);
}

struct isl_obj_vtable isl_obj_union_pw_qpolynomial_vtable = {
	isl_obj_union_pw_qp_copy,
	isl_obj_union_pw_qp_add,
	isl_obj_union_pw_qp_print,
	isl_obj_union_pw_qp_free
};

static void *isl_obj_pw_qpf_copy(void *v)
{
	return isl_pw_qpolynomial_fold_copy((struct isl_pw_qpolynomial_fold *)v);
}

static void isl_obj_pw_qpf_free(void *v)
{
	isl_pw_qpolynomial_fold_free((struct isl_pw_qpolynomial_fold *)v);
}

static __isl_give isl_printer *isl_obj_pw_qpf_print(__isl_take isl_printer *p,
	void *v)
{
	return isl_printer_print_pw_qpolynomial_fold(p,
					(struct isl_pw_qpolynomial_fold *)v);
}

static void *isl_obj_pw_qpf_add(void *v1, void *v2)
{
	return isl_pw_qpolynomial_fold_fold((struct isl_pw_qpolynomial_fold *)v1,
					    (struct isl_pw_qpolynomial_fold *)v2);
}

struct isl_obj_vtable isl_obj_pw_qpolynomial_fold_vtable = {
	isl_obj_pw_qpf_copy,
	isl_obj_pw_qpf_add,
	isl_obj_pw_qpf_print,
	isl_obj_pw_qpf_free
};

static void *isl_obj_union_pw_qpf_copy(void *v)
{
	return isl_union_pw_qpolynomial_fold_copy((struct isl_union_pw_qpolynomial_fold *)v);
}

static void isl_obj_union_pw_qpf_free(void *v)
{
	isl_union_pw_qpolynomial_fold_free((struct isl_union_pw_qpolynomial_fold *)v);
}

static __isl_give isl_printer *isl_obj_union_pw_qpf_print(
	__isl_take isl_printer *p, void *v)
{
	return isl_printer_print_union_pw_qpolynomial_fold(p,
				    (struct isl_union_pw_qpolynomial_fold *)v);
}

static void *isl_obj_union_pw_qpf_add(void *v1, void *v2)
{
	return isl_union_pw_qpolynomial_fold_fold(
				    (struct isl_union_pw_qpolynomial_fold *)v1,
				    (struct isl_union_pw_qpolynomial_fold *)v2);
}

struct isl_obj_vtable isl_obj_union_pw_qpolynomial_fold_vtable = {
	isl_obj_union_pw_qpf_copy,
	isl_obj_union_pw_qpf_add,
	isl_obj_union_pw_qpf_print,
	isl_obj_union_pw_qpf_free
};

static void *isl_obj_schedule_copy(void *v)
{
	return isl_schedule_copy((isl_schedule *) v);
}

static void isl_obj_schedule_free(void *v)
{
	isl_schedule_free((isl_schedule *) v);
}

static __isl_give isl_printer *isl_obj_schedule_print(
	__isl_take isl_printer *p, void *v)
{
	return isl_printer_print_schedule(p, (isl_schedule *) v);
}

struct isl_obj_vtable isl_obj_schedule_vtable = {
	isl_obj_schedule_copy,
	NULL,
	isl_obj_schedule_print,
	isl_obj_schedule_free
};
