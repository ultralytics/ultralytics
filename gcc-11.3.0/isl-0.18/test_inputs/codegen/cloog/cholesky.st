domain: "[n] -> { S2[i0, i1] : i0 >= 1 and i0 <= n and i1 >= 1 and i1 <= -1 + i0; S1[i0] : i0 >= 1 and i0 <= n; S4[i0, i1] : i0 >= 1 and i0 <= n and i1 >= 1 + i0 and i1 <= n; S5[i0, i1, i2] : i0 >= 1 and i0 <= n and i1 >= 1 + i0 and i1 <= n and i2 >= 1 and i2 <= -1 + i0; S6[i0, i1] : i0 >= 1 and i0 <= n and i1 >= 1 + i0 and i1 <= n; S3[i0] : i0 >= 1 and i0 <= n }"
child:
  context: "[n] -> { [] }"
  child:
    schedule: "[n] -> [{ S1[i0] -> [(i0)]; S4[i0, i1] -> [(i0)]; S6[i0, i1] -> [(i0)]; S3[i0] -> [(i0)]; S5[i0, i1, i2] -> [(i0)]; S2[i0, i1] -> [(i0)] }]"
    options: "[n] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[n] -> { S1[i0] }"
      - filter: "[n] -> { S2[i0, i1] }"
        child:
          schedule: "[n] -> [{ S2[i0, i1] -> [(i1)] }]"
          options: "[n] -> { separate[i0] }"
      - filter: "[n] -> { S3[i0] }"
      - filter: "[n] -> { S4[i0, i1]; S5[i0, i1, i2]; S6[i0, i1] }"
        child:
          schedule: "[n] -> [{ S4[i0, i1] -> [(i1)]; S6[i0, i1] -> [(i1)]; S5[i0, i1, i2] -> [(i1)] }]"
          options: "[n] -> { separate[i0] }"
          child:
            sequence:
            - filter: "[n] -> { S4[i0, i1] }"
            - filter: "[n] -> { S5[i0, i1, i2] }"
              child:
                schedule: "[n] -> [{ S5[i0, i1, i2] -> [(i2)] }]"
                options: "[n] -> { separate[i0] }"
            - filter: "[n] -> { S6[i0, i1] }"
