domain: "[M] -> { S1[2M, M] }"
child:
  context: "[M] -> { [] }"
  child:
    schedule: "[M] -> [{ S1[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)] }]"
    options: "[M] -> { separate[i0] }"
