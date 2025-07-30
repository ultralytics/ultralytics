#ifndef ISL_YAML_H
#define ISL_YAML_H

#define ISL_YAML_INDENT_FLOW		-1

enum isl_yaml_state {
	isl_yaml_none,
	isl_yaml_mapping_first_key_start,
	isl_yaml_mapping_key_start,
	isl_yaml_mapping_key,
	isl_yaml_mapping_val_start,
	isl_yaml_mapping_val,
	isl_yaml_sequence_first_start,
	isl_yaml_sequence_start,
	isl_yaml_sequence
};

#endif
