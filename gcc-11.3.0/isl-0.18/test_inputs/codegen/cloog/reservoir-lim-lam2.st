domain: "[M, N] -> { S3[i0, i1] : i0 >= 1 and i0 <= M and i1 >= 1 and i1 <= -1 + N; S1[i0] : i0 >= 1 and i0 <= M; S2[i0, i1] : i0 >= 1 and i0 <= M and i1 >= 2 and i1 <= N }"
child:
  context: "[M, N] -> { [] : M >= 1 and N >= 1 }"
  child:
    sequence:
    - filter: "[M, N] -> { S1[i0] }"
      child:
        schedule: "[M, N] -> [{ S1[i0] -> [(i0)] }]"
        options: "[M, N] -> { separate[i0] }"
    - filter: "[M, N] -> { S2[i0, i1] }"
      child:
        schedule: "[M, N] -> [{ S2[i0, i1] -> [(i0)] }]"
        options: "[M, N] -> { separate[i0] }"
        child:
          schedule: "[M, N] -> [{ S2[i0, i1] -> [(i1)] }]"
          options: "[M, N] -> { separate[i0] }"
    - filter: "[M, N] -> { S3[i0, i1] }"
      child:
        schedule: "[M, N] -> [{ S3[i0, i1] -> [(i0)] }]"
        options: "[M, N] -> { separate[i0] }"
        child:
          schedule: "[M, N] -> [{ S3[i0, i1] -> [(i1)] }]"
          options: "[M, N] -> { separate[i0] }"
