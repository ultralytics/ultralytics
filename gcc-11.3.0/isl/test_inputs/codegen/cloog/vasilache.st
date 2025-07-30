domain: "[M, N] -> { S5[i0, i1] : i0 >= 0 and i0 <= -1 + N and i1 >= 0 and i1 <= -1 + N; S8[]; S2[]; S7[i0, i1, i2, i3] : i0 >= 0 and i0 <= -1 + N and i1 >= 0 and i1 <= -1 + N and i3 >= 0 and i3 <= -1 + N and i3 >= 32i2 and i3 <= 31 + 32i2; S4[i0, i1] : i0 >= 0 and i0 <= -1 + N and i1 >= 0 and i1 <= -1 + N; S1[]; S3[] : M >= 79; S6[i0, i1, i2, i3] : i0 >= 0 and i0 <= -1 + N and i1 >= 0 and i1 <= -1 + N and i3 >= 0 and i3 <= -1 + N and i3 >= 32i2 and i3 <= 31 + 32i2 }"
child:
  context: "[M, N] -> { [] : M <= 3 and N >= 100 }"
  child:
    sequence:
    - filter: "[M, N] -> { S1[] }"
    - filter: "[M, N] -> { S2[] }"
    - filter: "[M, N] -> { S3[] }"
    - filter: "[M, N] -> { S5[i0, i1]; S4[i0, i1] }"
      child:
        schedule: "[M, N] -> [{ S5[i0, i1] -> [(i0)]; S4[i0, i1] -> [(i0)] }]"
        options: "[M, N] -> { separate[i0] }"
        child:
          schedule: "[M, N] -> [{ S5[i0, i1] -> [(i1)]; S4[i0, i1] -> [(i1)] }]"
          options: "[M, N] -> { separate[i0] }"
          child:
            sequence:
            - filter: "[M, N] -> { S4[i0, i1] }"
            - filter: "[M, N] -> { S5[i0, i1] }"
    - filter: "[M, N] -> { S7[i0, i1, i2, i3]; S6[i0, i1, i2, i3] }"
      child:
        schedule: "[M, N] -> [{ S7[i0, i1, i2, i3] -> [(i0)]; S6[i0, i1, i2, i3] -> [(i0)] }]"
        options: "[M, N] -> { separate[i0] }"
        child:
          schedule: "[M, N] -> [{ S7[i0, i1, i2, i3] -> [(i1)]; S6[i0, i1, i2, i3] -> [(i1)] }]"
          options: "[M, N] -> { separate[i0] }"
          child:
            schedule: "[M, N] -> [{ S7[i0, i1, i2, i3] -> [(i2)]; S6[i0, i1, i2, i3] -> [(i2)] }]"
            options: "[M, N] -> { separate[i0] }"
            child:
              schedule: "[M, N] -> [{ S7[i0, i1, i2, i3] -> [(i3)]; S6[i0, i1, i2, i3] -> [(1 + i3)] }]"
              options: "[M, N] -> { separate[i0] }"
              child:
                sequence:
                - filter: "[M, N] -> { S6[i0, i1, i2, i3] }"
                - filter: "[M, N] -> { S7[i0, i1, i2, i3] }"
    - filter: "[M, N] -> { S8[] }"
