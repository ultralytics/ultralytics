domain: "[M, N] -> { S3[i0, i1] : i0 >= 0 and i0 <= M and i1 >= 0 and i1 <= N; S1[i0, i1] : i0 >= 0 and i0 <= M and i1 >= 0 and i1 <= N; S2[i0, i1] : i0 >= 0 and i0 <= M and i1 >= 0 and i1 <= N }"
child:
  context: "[M, N] -> { [] }"
  child:
    schedule: "[M, N] -> [{ S1[i0, i1] -> [(-4 + 3i0 + i1)]; S2[i0, i1] -> [(3i0 + i1)]; S3[i0, i1] -> [(3i0 + i1)] }]"
    options: "[M, N] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M, N] -> { S1[i0, i1]; S2[i0, i1] }"
        child:
          schedule: "[M, N] -> [{ S1[i0, i1] -> [(i1)]; S2[i0, i1] -> [(i1)] }]"
          options: "[M, N] -> { separate[i0] }"
      - filter: "[M, N] -> { S3[i0, i1] }"
        child:
          schedule: "[M, N] -> [{ S3[i0, i1] -> [(i1)] }]"
          options: "[M, N] -> { separate[i0] }"
