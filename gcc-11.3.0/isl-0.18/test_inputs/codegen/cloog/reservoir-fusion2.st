domain: "[M, N] -> { S1[i0, i1] : i0 >= 1 and i0 <= N and i1 >= 1 and i1 <= M; S2[i0, i1] : i0 >= 1 and i0 <= N and i1 >= 1 and i1 <= M }"
child:
  context: "[M, N] -> { [] }"
  child:
    schedule: "[M, N] -> [{ S1[i0, i1] -> [(i0)]; S2[i0, i1] -> [(1 + i0)] }]"
    options: "[M, N] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M, N] -> { S2[i0, i1] }"
        child:
          schedule: "[M, N] -> [{ S2[i0, i1] -> [(i1)] }]"
          options: "[M, N] -> { separate[i0] }"
      - filter: "[M, N] -> { S1[i0, i1] }"
        child:
          schedule: "[M, N] -> [{ S1[i0, i1] -> [(i1)] }]"
          options: "[M, N] -> { separate[i0] }"
