domain: "[M, N] -> { S1[i0, i1] : (i0 >= M and i0 <= N and i1 >= 1 and i1 <= M) or (i0 >= 1 and i0 <= N and i0 <= 2M and i1 >= 1 and i1 <= M) }"
child:
  context: "[M, N] -> { [] }"
  child:
    schedule: "[M, N] -> [{ S1[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)] }]"
    options: "[M, N] -> { separate[i0] }"
