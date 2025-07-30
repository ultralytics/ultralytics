#ifndef ISL_AST_BUILD_PRIVATE_H
#define ISL_AST_BUILD_PRIVATE_H

#include <isl/aff.h>
#include <isl/ast.h>
#include <isl/ast_build.h>
#include <isl/set.h>
#include <isl/list.h>
#include <isl/schedule_node.h>

/* An isl_ast_build represents the context in which AST is being
 * generated.  That is, it (mostly) contains information about outer
 * loops that can be used to simplify inner loops.
 *
 * "domain" represents constraints on the internal schedule domain,
 * corresponding to the context of the AST generation and the constraints
 * implied by the loops that have already been generated.
 * When an isl_ast_build is first created, outside any AST generation,
 * the domain is typically a parameter set.  It is only when a AST
 * generation phase is initiated that the domain of the isl_ast_build
 * is changed to refer to the internal schedule domain.
 * The domain then lives in a space of the form
 *
 *	S
 *
 *  or
 *
 *	[O -> S]
 *
 * O represents the loops generated in outer AST generations.
 * S represents the loops (both generated and to be generated)
 * of the current AST generation.
 * Both include eliminated loops.
 * "domain" is expected not to have any unknown divs because
 * it is used as the context argument in a call to isl_basic_set_gist
 * in isl_ast_build_compute_gist_basic_set.
 *
 * "depth" is equal to the number of loops that have already
 * been generated (including those in outer AST generations).
 * "outer_pos" is equal to the number of loops in outer AST generations.
 *
 * "generated" is a superset of "domain" corresponding to those
 * constraints that were either given by the user or that have
 * effectively been generated (as bounds on a for loop).
 *
 * "pending" is a superset of "domain" corresponding to the constraints
 * that still need to be generated (as guards), but that may end up
 * not getting generated if they are implied by any constraints
 * enforced by inner loops.
 *
 * "strides" contains the stride of each loop.  The number of elements
 * is equal to the number of dimensions in "domain".
 * "offsets" constains the offsets of strided loops.  If s is the stride
 * for a given dimension and f is the corresponding offset, then the
 * dimension takes on values
 *
 *	f + s a
 *
 * with a an integer.  For non-strided loops, the offset is zero.
 *
 * "iterators" contains the loop iterators of both generated and
 * to be generated loops.  The number of elements is at least as
 * large as the dimension of the internal schedule domain.  The
 * number may be larger, in which case the additional ids can be
 * used in a nested AST generation should the schedule be non-injective.
 *
 * "values" lives in the space
 *
 *	[O -> S] -> [O -> S]		(or S -> S)
 *
 * and expresses (if possible) loop iterators in terms of parameters
 * and outer loop iterators.  If the value of a given loop iterator
 * cannot be expressed as an affine expression (either because the iterator
 * attains multiple values or because the single value is a piecewise
 * affine expression), then it is expressed in "values" as being equal
 * to itself.
 *
 * "value" is the value of the loop iterator at the current depth.
 * It is NULL if it has not been computed yet or if the value of the
 * given loop iterator cannot be expressed as a piecewise affine expression
 * (because the iterator attains multiple values).
 *
 * "schedule_map" maps the internal schedule domain to the external schedule
 * domain.  It may be NULL if it hasn't been computed yet.
 * See isl_ast_build_get_schedule_map_multi_aff.
 *
 * "internal2input" maps the internal schedule domain to the original
 * input schedule domain.  In case of a schedule tree input, the original
 * input schedule domain consist of the flat product of all outer
 * band node spaces, including the current band node.
 * It may be NULL if there no longer is such a uniform mapping
 * (because different iterations have been rescheduled differently).
 *
 * "options" contains the AST build options in case we are generating
 * an AST from a flat schedule map.  When creating an AST from a schedule
 * tree, this field is ignored.
 *
 * The "create_leaf" callback is called for every leaf in the generated AST.
 * The callback is responsible for creating the node to be placed at those
 * leaves.  If this callback is not set, then isl will generated user
 * nodes with call expressions corresponding to an element of the domain.
 *
 * The "at_each_domain" callback is called on every node created to represent
 * an element of the domain.  Each of these nodes is a user node
 * with as expression a call expression.
 *
 * The "before_each_for" callback is called on each for node before
 * its children have been created.
 *
 * The "after_each_for" callback is called on each for node after
 * its children have been created.
 *
 * The "before_each_mark" callback is called before we handle the subtree
 * of an isl_schedule_node_mark node.
 *
 * The "after_each_mark" callback is called after we have handled the subtree
 * of an isl_schedule_node_mark node.
 *
 * "executed" contains the inverse schedule at this point
 * of the AST generation.
 * It is currently only used in isl_ast_build_get_schedule, which is
 * in turn only used by user code from within a callback.
 * The value is set right before we may be calling such a callback.
 *
 * "single_valued" is set if the current inverse schedule (which may or may
 * not be stored in "executed") is known to be single valued, specifically
 * an inverse schedule that was not (appeared not to be) single valued
 * is extended to a single valued inverse schedule.  This is mainly used
 * to avoid an infinite recursion when we fail to detect later on that
 * the extended inverse schedule is single valued.
 *
 * "node" points to the current band node in case we are generating
 * an AST from a schedule tree.  It may be NULL if we are not generating
 * an AST from a schedule tree or if we are not inside a band node.
 *
 * "loop_type" originally constains loop AST generation types for
 * the "n" members of "node" and it is updated (along with "n") when
 * a schedule dimension is inserted.
 * It is NULL if "node" is NULL.
 *
 * "isolated" is the piece of the schedule domain isolated by the isolate
 * option on the current band.  This set may be NULL if we have not checked
 * for the isolate option yet.
 */
struct isl_ast_build {
	int ref;

	int outer_pos;
	int depth;

	isl_id_list *iterators;

	isl_set *domain;
	isl_set *generated;
	isl_set *pending;
	isl_multi_aff *values;

	isl_pw_aff *value;

	isl_vec *strides;
	isl_multi_aff *offsets;

	isl_multi_aff *schedule_map;
	isl_multi_aff *internal2input;

	isl_union_map *options;

	__isl_give isl_ast_node *(*at_each_domain)(
		__isl_take isl_ast_node *node,
		__isl_keep isl_ast_build *build, void *user);
	void *at_each_domain_user;

	__isl_give isl_id *(*before_each_for)(
		__isl_keep isl_ast_build *context, void *user);
	void *before_each_for_user;
	__isl_give isl_ast_node *(*after_each_for)(
		__isl_take isl_ast_node *node,
		__isl_keep isl_ast_build *context, void *user);
	void *after_each_for_user;

	isl_stat (*before_each_mark)(__isl_keep isl_id *mark,
		__isl_keep isl_ast_build *build, void *user);
	void *before_each_mark_user;
	__isl_give isl_ast_node *(*after_each_mark)(
		__isl_take isl_ast_node *node,
		__isl_keep isl_ast_build *context, void *user);
	void *after_each_mark_user;

	__isl_give isl_ast_node *(*create_leaf)(
		__isl_take isl_ast_build *build, void *user);
	void *create_leaf_user;

	isl_union_map *executed;
	int single_valued;

	isl_schedule_node *node;
	int n;
	enum isl_ast_loop_type *loop_type;
	isl_set *isolated;
};

__isl_give isl_ast_build *isl_ast_build_clear_local_info(
	__isl_take isl_ast_build *build);
__isl_give isl_ast_build *isl_ast_build_increase_depth(
	__isl_take isl_ast_build *build);
int isl_ast_build_get_depth(__isl_keep isl_ast_build *build);
unsigned isl_ast_build_dim(__isl_keep isl_ast_build *build,
	enum isl_dim_type type);
__isl_give isl_space *isl_ast_build_get_space(
	__isl_keep isl_ast_build *build, int internal);
__isl_give isl_ast_build *isl_ast_build_align_params(
	__isl_take isl_ast_build *build, __isl_take isl_space *model);
__isl_give isl_ast_build *isl_ast_build_cow(
	__isl_take isl_ast_build *build);
__isl_give isl_ast_build *isl_ast_build_insert_dim(
	__isl_take isl_ast_build *build, int pos);
__isl_give isl_ast_build *isl_ast_build_scale_down(
	__isl_take isl_ast_build *build, __isl_take isl_val *m,
	__isl_take isl_union_map *umap);
__isl_give isl_ast_build *isl_ast_build_product(
	__isl_take isl_ast_build *build, __isl_take isl_space *embedding);
__isl_give isl_ast_build *isl_ast_build_set_loop_bounds(
	__isl_take isl_ast_build *build, __isl_take isl_basic_set *bounds);
__isl_give isl_ast_build *isl_ast_build_set_pending_generated(
	__isl_take isl_ast_build *build, __isl_take isl_basic_set *bounds);
__isl_give isl_ast_build *isl_ast_build_detect_strides(
	__isl_take isl_ast_build *build, __isl_take isl_set *set);
__isl_give isl_ast_build *isl_ast_build_include_stride(
	__isl_take isl_ast_build *build);
__isl_give isl_ast_build *isl_ast_build_set_executed(
	__isl_take isl_ast_build *build,
	__isl_take isl_union_map *executed);
__isl_give isl_ast_build *isl_ast_build_set_single_valued(
	__isl_take isl_ast_build *build, int sv);
__isl_give isl_multi_aff *isl_ast_build_get_internal2input(
	__isl_keep isl_ast_build *build);
__isl_give isl_set *isl_ast_build_get_domain(
	__isl_keep isl_ast_build *build);
__isl_give isl_set *isl_ast_build_get_pending(
	__isl_keep isl_ast_build *build);
__isl_give isl_set *isl_ast_build_get_generated(
	__isl_keep isl_ast_build *build);
__isl_give isl_ast_build *isl_ast_build_restrict_generated(
	__isl_take isl_ast_build *build, __isl_take isl_set *set);
__isl_give isl_ast_build *isl_ast_build_replace_pending_by_guard(
	__isl_take isl_ast_build *build, __isl_take isl_set *guard);
__isl_give int isl_ast_build_need_schedule_map(
	__isl_keep isl_ast_build *build);
__isl_give isl_multi_aff *isl_ast_build_get_schedule_map_multi_aff(
	__isl_keep isl_ast_build *build);
__isl_give isl_map *isl_ast_build_get_schedule_map(
	__isl_keep isl_ast_build *build);
int isl_ast_build_has_affine_value(__isl_keep isl_ast_build *build, int pos);
int isl_ast_build_has_value(__isl_keep isl_ast_build *build);
__isl_give isl_id *isl_ast_build_get_iterator_id(
	__isl_keep isl_ast_build *build, int pos);

int isl_ast_build_has_schedule_node(__isl_keep isl_ast_build *build);
__isl_give isl_schedule_node *isl_ast_build_get_schedule_node(
	__isl_keep isl_ast_build *build);
__isl_give isl_ast_build *isl_ast_build_set_schedule_node(
	__isl_take isl_ast_build *build,
	__isl_take isl_schedule_node *node);
__isl_give isl_ast_build *isl_ast_build_reset_schedule_node(
	__isl_take isl_ast_build *build);

__isl_give isl_ast_build *isl_ast_build_extract_isolated(
	__isl_take isl_ast_build *build);
int isl_ast_build_has_isolated(__isl_keep isl_ast_build *build);
__isl_give isl_set *isl_ast_build_get_isolated(
	__isl_keep isl_ast_build *build);

__isl_give isl_basic_set *isl_ast_build_specialize_basic_set(
	__isl_keep isl_ast_build *build, __isl_take isl_basic_set *bset);
__isl_give isl_basic_set *isl_ast_build_compute_gist_basic_set(
	__isl_keep isl_ast_build *build, __isl_take isl_basic_set *bset);
__isl_give isl_set *isl_ast_build_specialize(__isl_keep isl_ast_build *build,
	__isl_take isl_set *set);
__isl_give isl_set *isl_ast_build_compute_gist(
	__isl_keep isl_ast_build *build, __isl_take isl_set *set);
__isl_give isl_map *isl_ast_build_compute_gist_map_domain(
	__isl_keep isl_ast_build *build, __isl_take isl_map *map);
__isl_give isl_aff *isl_ast_build_compute_gist_aff(
	__isl_keep isl_ast_build *build, __isl_take isl_aff *aff);
__isl_give isl_pw_aff *isl_ast_build_compute_gist_pw_aff(
	__isl_keep isl_ast_build *build, __isl_take isl_pw_aff *pa);
__isl_give isl_pw_multi_aff *isl_ast_build_compute_gist_pw_multi_aff(
	__isl_keep isl_ast_build *build, __isl_take isl_pw_multi_aff *pma);

__isl_give isl_union_map *isl_ast_build_substitute_values_union_map_domain(
	__isl_keep isl_ast_build *build, __isl_take isl_union_map *umap);

int isl_ast_build_aff_is_nonneg(__isl_keep isl_ast_build *build,
	__isl_keep isl_aff *aff);

isl_bool isl_ast_build_has_stride(__isl_keep isl_ast_build *build, int pos);
__isl_give isl_aff *isl_ast_build_get_offset(__isl_keep isl_ast_build *build,
	int pos);
__isl_give isl_val *isl_ast_build_get_stride(__isl_keep isl_ast_build *build,
	int pos);
__isl_give isl_set *isl_ast_build_get_stride_constraint(
	__isl_keep isl_ast_build *build);
__isl_give isl_multi_aff *isl_ast_build_get_stride_expansion(
	__isl_keep isl_ast_build *build);

void isl_ast_build_dump(__isl_keep isl_ast_build *build);

__isl_give isl_set *isl_ast_build_get_option_domain(
	__isl_keep isl_ast_build *build, enum isl_ast_loop_type type);
__isl_give isl_map *isl_ast_build_get_separation_class(
	__isl_keep isl_ast_build *build);
__isl_give isl_set *isl_ast_build_eliminate(
	__isl_keep isl_ast_build *build, __isl_take isl_set *domain);
__isl_give isl_set *isl_ast_build_eliminate_inner(
	__isl_keep isl_ast_build *build, __isl_take isl_set *set);
__isl_give isl_set *isl_ast_build_eliminate_divs(
	__isl_keep isl_ast_build *build, __isl_take isl_set *set);

enum isl_ast_loop_type isl_ast_build_get_loop_type(
	__isl_keep isl_ast_build *build, int isolated);

__isl_give isl_map *isl_ast_build_map_to_iterator(
	__isl_keep isl_ast_build *build, __isl_take isl_set *set);

int isl_ast_build_options_involve_depth(__isl_keep isl_ast_build *build);

#endif
