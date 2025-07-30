#include <isl/set.h>
#include <isl/vertices.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct isl_morph;

/* A parametric vertex.  "vertex" contains the actual description
 * of the vertex as a singleton parametric set.  "dom" is the projection
 * of "vertex" onto the parameter space, i.e., the activity domain
 * of the vertex.
 * During the construction of vertices and chambers, the activity domain
 * of every parametric vertex is full-dimensional.
 */
struct isl_vertex {
	isl_basic_set *dom;
	isl_basic_set *vertex;
};

/* A chamber in the chamber decomposition.  The indices of the "n_vertices"
 * active vertices are stored in "vertices".
 */
struct isl_chamber {
	int n_vertices;
	int *vertices;
	isl_basic_set *dom;
};

struct isl_vertices {
	int ref;

	/* The rational basic set spanned by the vertices. */
	isl_basic_set *bset;

	int n_vertices;
	struct isl_vertex *v;

	int n_chambers;
	struct isl_chamber *c;
};

struct isl_cell {
	int n_vertices;
	int *ids;
	isl_vertices *vertices;
	isl_basic_set *dom;
};

struct isl_external_vertex {
	isl_vertices *vertices;
	int id;
};

int isl_vertices_foreach_disjoint_cell(__isl_keep isl_vertices *vertices,
	int (*fn)(__isl_take isl_cell *cell, void *user), void *user);
int isl_cell_foreach_simplex(__isl_take isl_cell *cell,
	int (*fn)(__isl_take isl_cell *simplex, void *user), void *user);

__isl_give isl_vertices *isl_morph_vertices(__isl_take struct isl_morph *morph,
	__isl_take isl_vertices *vertices);

#if defined(__cplusplus)
}
#endif
