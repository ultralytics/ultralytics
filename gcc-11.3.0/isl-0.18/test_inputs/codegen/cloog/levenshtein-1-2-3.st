domain: "[M, N] -> { S5[i0, i1] : 2i1 = -N + i0 and i0 >= 2 + N and i0 <= -2 + 2M - N and N >= 1; S3[i0, i0] : i0 >= 1 and i0 <= N and N <= -2 + M; S7[i0, i1] : i0 >= 1 + N and 2i1 <= -1 - N + i0 and i0 <= -2 + 2M - N and 2i1 >= -2 - N + i0 and N <= -2 + M and N >= 1; S6[i0, i1] : 2i1 <= -1 + N + i0 and i1 <= -1 + i0 and i1 >= 1 - M + i0 and 2i1 >= 1 - N + i0 and i1 >= 1 and i1 <= -1 + M and N <= -2 + M; S1[0, 0] : N <= -2 + M and N >= 1; S2[i0, 0] : i0 >= 1 and i0 <= N and N <= -2 + M; S4[i0, i1] : 2i1 = N + i0 and i0 >= 2 + N and i0 <= -2 + 2M - N and N >= 1; S8[i0, i1] : i0 >= 1 + N and 2i1 <= N + i0 and 2i1 >= -N + i0 and i0 <= -2 + 2M - N and N <= -2 + M and N >= 1 }"
child:
  context: "[M, N] -> { [] : N <= -2 + M and N >= 1 }"
  child:
    schedule: "[M, N] -> [{ S7[i0, i1] -> [(i0)]; S5[i0, i1] -> [(i0)]; S1[i0, i1] -> [(i0)]; S3[i0, i1] -> [(i0)]; S2[i0, i1] -> [(i0)]; S4[i0, i1] -> [(i0)]; S8[i0, i1] -> [(i0)]; S6[i0, i1] -> [(i0)] }, { S7[i0, i1] -> [(i1)]; S5[i0, i1] -> [(i1)]; S1[i0, i1] -> [(i1)]; S3[i0, i1] -> [(i1)]; S2[i0, i1] -> [(i1)]; S4[i0, i1] -> [(i1)]; S8[i0, i1] -> [(i1)]; S6[i0, i1] -> [(i1)] }]"
    options: "[M, N] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M, N] -> { S1[i0, i1] }"
      - filter: "[M, N] -> { S2[i0, i1] }"
      - filter: "[M, N] -> { S3[i0, i1] }"
      - filter: "[M, N] -> { S4[i0, i1] }"
      - filter: "[M, N] -> { S5[i0, i1] }"
      - filter: "[M, N] -> { S6[i0, i1] }"
      - filter: "[M, N] -> { S7[i0, i1] }"
      - filter: "[M, N] -> { S8[i0, i1] }"
