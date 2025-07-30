domain: "[M] -> { S2[i0, i1, i2] : i0 >= 1 and i0 <= -1 + M and i1 >= 0 and i2 >= 1 + i1 and i2 <= -1 + M; S1[i0, i1, i2] : i0 >= 1 and i0 <= -1 + M and i1 >= 0 and i2 >= 0 and i2 <= -1 + M - i1 }"
child:
  context: "[M] -> { [] }"
  child:
    schedule: "[M] -> [{ S1[i0, i1, i2] -> [(i0 + i1 + i2)]; S2[i0, i1, i2] -> [(i0 + i2)] }]"
    options: "[M] -> { separate[i0] }"
    child:
      schedule: "[M] -> [{ S1[i0, i1, i2] -> [(-i2)]; S2[i0, i1, i2] -> [(i1 - i2)] }]"
      options: "[M] -> { separate[i0] }"
      child:
        sequence:
        - filter: "[M] -> { S1[i0, i1, i2] }"
          child:
            schedule: "[M] -> [{ S1[i0, i1, i2] -> [(i2)] }]"
            options: "[M] -> { separate[i0] }"
        - filter: "[M] -> { S2[i0, i1, i2] }"
          child:
            schedule: "[M] -> [{ S2[i0, i1, i2] -> [(i2)] }]"
            options: "[M] -> { separate[i0] }"
