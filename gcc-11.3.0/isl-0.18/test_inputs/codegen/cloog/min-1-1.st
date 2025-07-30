domain: "[M, N] -> { S1[i0, i1] : i0 >= 1 and i1 >= 0 and i1 <= M and i1 <= i0 and i1 <= N - i0 }"
child:
  context: "[M, N] -> { [] }"
  child:
    schedule: "[M, N] -> [{ S1[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)] }]"
    options: "[M, N] -> { separate[i0] }"
