domain: "[M, N] -> { S1[2M, 2 + N] }"
child:
  context: "[M, N] -> { [] }"
  child:
    schedule: "[M, N] -> [{ S1[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)] }]"
    options: "[M, N] -> { separate[i0] }"
