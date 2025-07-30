domain: "[M, N] -> { S5[i0] : i0 >= 0 and i0 <= -1 + N; S1[i0] : i0 >= 0 and i0 <= -1 + N; S3[i0] : i0 >= 0 and i0 <= -1 + N; S2[i0, i1] : i0 >= 0 and i0 <= -1 + N and i1 >= i0 and i1 <= -1 + M; S6[i0, i1] : i0 >= 0 and i1 >= 1 + i0 and i1 <= -1 + N; S9[i0, i1, i2] : i0 >= 0 and i1 >= 1 + i0 and i1 <= -1 + N and i2 >= i0 and i2 <= -1 + M; S4[i0, i1] : i0 >= 0 and i0 <= -1 + N and i1 >= i0 and i1 <= -1 + M; S8[i0, i1] : i0 >= 0 and i1 >= 1 + i0 and i1 <= -1 + N; S10[i0] : i0 >= 0 and i0 <= -1 + N; S7[i0, i1, i2] : i0 >= 0 and i1 >= 1 + i0 and i1 <= -1 + N and i2 >= i0 and i2 <= -1 + M }"
child:
  context: "[M, N] -> { [] }"
  child:
    schedule: "[M, N] -> [{ S3[i0] -> [(1 + i0)]; S10[i0] -> [(1 + i0)]; S5[i0] -> [(1 + i0)]; S7[i0, i1, i2] -> [(2 + i0)]; S9[i0, i1, i2] -> [(2 + i0)]; S2[i0, i1] -> [(1 + i0)]; S4[i0, i1] -> [(1 + i0)]; S8[i0, i1] -> [(2 + i0)]; S1[i0] -> [(i0)]; S6[i0, i1] -> [(2 + i0)] }]"
    options: "[M, N] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M, N] -> { S6[i0, i1]; S9[i0, i1, i2]; S8[i0, i1]; S7[i0, i1, i2] }"
        child:
          schedule: "[M, N] -> [{ S7[i0, i1, i2] -> [(i1)]; S9[i0, i1, i2] -> [(i1)]; S8[i0, i1] -> [(i1)]; S6[i0, i1] -> [(i1)] }]"
          options: "[M, N] -> { separate[i0] }"
          child:
            sequence:
            - filter: "[M, N] -> { S6[i0, i1] }"
            - filter: "[M, N] -> { S7[i0, i1, i2] }"
              child:
                schedule: "[M, N] -> [{ S7[i0, i1, i2] -> [(i2)] }]"
                options: "[M, N] -> { separate[i0] }"
            - filter: "[M, N] -> { S8[i0, i1] }"
            - filter: "[M, N] -> { S9[i0, i1, i2] }"
              child:
                schedule: "[M, N] -> [{ S9[i0, i1, i2] -> [(i2)] }]"
                options: "[M, N] -> { separate[i0] }"
      - filter: "[M, N] -> { S2[i0, i1] }"
        child:
          schedule: "[M, N] -> [{ S2[i0, i1] -> [(i1)] }]"
          options: "[M, N] -> { separate[i0] }"
      - filter: "[M, N] -> { S3[i0] }"
      - filter: "[M, N] -> { S4[i0, i1] }"
        child:
          schedule: "[M, N] -> [{ S4[i0, i1] -> [(i1)] }]"
          options: "[M, N] -> { separate[i0] }"
      - filter: "[M, N] -> { S10[i0] }"
      - filter: "[M, N] -> { S1[i0] }"
      - filter: "[M, N] -> { S5[i0] }"
