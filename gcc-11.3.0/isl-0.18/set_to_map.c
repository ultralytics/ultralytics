#include <isl/map_type.h>

/* Treat "set" as a map.
 * Internally, isl_set is defined to isl_map, so in practice,
 * this function performs a redundant cast.
 */
static __isl_give isl_map *set_to_map(__isl_take isl_set *set)
{
	return (isl_map *) set;
}
